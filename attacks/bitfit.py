import os, csv, time, random, logging, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

DATA_ROOT = "./datasets"
OUT_DIR = "./results_clean/bitfit"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 64
WORKERS = 4
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_CLASSES = 10

RE_LABEL = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]

CHECKPOINTS = [
    {"method":"SOPHON-IL","name":"sophon_il_cifar10","type":"sophon","arch":"resnet18","dataset":"CIFAR10","task":"CIFAR10","path":"./checkpoints/sophon_il/sophon_il_cifar10.pth"},
    {"method":"SOPHON-IL","name":"sophon_il_cinic","type":"sophon","arch":"resnet18","dataset":"CINIC","task":"CINIC","path":"./checkpoints/sophon_il/sophon_il_cinic.pth"},
    {"method":"SOPHON-IL","name":"sophon_il_mnist","type":"sophon","arch":"resnet18","dataset":"MNIST","task":"MNIST","path":"./checkpoints/sophon_il/sophon_il_mnist.pth"},
    {"method":"SOPHON-IL","name":"sophon_il_stl10","type":"sophon","arch":"resnet18","dataset":"STL10","task":"STL10","path":"./checkpoints/sophon_il/sophon_il_stl10.pth"},
    {"method":"SOPHON-IL","name":"sophon_il_svhn","type":"sophon","arch":"resnet18","dataset":"SVHN","task":"SVHN","path":"./checkpoints/sophon_il/sophon_il_svhn.pth"},

    {"method":"SOPHON-KL","name":"sophon_kl_cifar10","type":"sophon","arch":"resnet18","dataset":"CIFAR10","task":"CIFAR10","path":"./checkpoints/sophon_kl/sophon_kl_cifar10.pth"},
    {"method":"SOPHON-KL","name":"sophon_kl_cinic","type":"sophon","arch":"resnet18","dataset":"CINIC","task":"CINIC","path":"./checkpoints/sophon_kl/sophon_kl_cinic.pth"},
    {"method":"SOPHON-KL","name":"sophon_kl_mnist","type":"sophon","arch":"resnet18","dataset":"MNIST","task":"MNIST","path":"./checkpoints/sophon_kl/sophon_kl_mnist.pth"},
    {"method":"SOPHON-KL","name":"sophon_kl_stl10","type":"sophon","arch":"resnet18","dataset":"STL10","task":"STL10","path":"./checkpoints/sophon_kl/sophon_kl_stl10.pth"},
    {"method":"SOPHON-KL","name":"sophon_kl_svhn","type":"sophon","arch":"resnet18","dataset":"SVHN","task":"SVHN","path":"./checkpoints/sophon_kl/sophon_kl_svhn.pth"},

    {"method":"HNTL","name":"hntl_cifar_stl_vgg13","type":"vgg","arch":"vgg13","dataset":"STL10","task":"CIFAR10_to_STL10","path":"./checkpoints/hntl_cifar_stl_vgg13.pth"},
    {"method":"CUPI","name":"cupi_cifar_stl_vgg13","type":"vgg","arch":"vgg13","dataset":"STL10","task":"CIFAR10_to_STL10","path":"./checkpoints/cupi_cifar_stl_vgg13.pth"},
    {"method":"CUTI","name":"cuti_cifar_stl_vgg19","type":"vgg","arch":"vgg19","dataset":"STL10","task":"CIFAR10_to_STL10","path":"./checkpoints/cuti_cifar_stl_vgg19.pth"},
]

CSV_HEADER = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "attack","epoch","epochs","lr","batch_size",
    "epoch_acc","epoch_loss","best_acc_so_far","best_epoch_so_far",
    "epoch_time_sec","cumulative_time_sec",
    "trainable_bias_params","total_params","trainable_percent",
    "trainable_bias_names","missing_keys","unexpected_keys"
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class STL10Mapped(datasets.STL10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, RE_LABEL[target]

def get_loaders(dataset):
    tf_rgb = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    tf_mnist = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    d = dataset.upper()

    if d == "CIFAR10":
        train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=tf_rgb)
        test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=tf_rgb)
    elif d == "MNIST":
        train_set = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tf_mnist)
        test_set = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=tf_mnist)
    elif d == "SVHN":
        train_set = datasets.SVHN(DATA_ROOT, split="train", download=False, transform=tf_rgb)
        test_set = datasets.SVHN(DATA_ROOT, split="test", download=False, transform=tf_rgb)
    elif d == "STL10":
        train_set = STL10Mapped(DATA_ROOT, split="train", download=False, transform=tf_rgb)
        test_set = STL10Mapped(DATA_ROOT, split="test", download=False, transform=tf_rgb)
    elif d == "CINIC":
        cinic_root = "./datasets/CINIC10"
        train_dir = os.path.join(cinic_root, "train")
        valid_dir = os.path.join(cinic_root, "valid")
        test_dir = os.path.join(cinic_root, "test")

        if os.path.exists(train_dir) and os.path.exists(valid_dir):
            train_set = datasets.ImageFolder(train_dir, transform=tf_rgb)
            test_set = datasets.ImageFolder(valid_dir, transform=tf_rgb)
        elif os.path.exists(train_dir) and os.path.exists(test_dir):
            train_set = datasets.ImageFolder(train_dir, transform=tf_rgb)
            test_set = datasets.ImageFolder(test_dir, transform=tf_rgb)
        else:
            raise FileNotFoundError(f"CINIC folders not found inside {cinic_root}")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=WORKERS, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True
    )

    return train_loader, test_loader

class CustomVGG(nn.Module):
    def __init__(self, arch, num_classes=10):
        super().__init__()

        if arch == "vgg13":
            base = models.vgg13(weights=None)
        elif arch == "vgg19":
            base = models.vgg19(weights=None)
        else:
            raise ValueError(f"Unknown VGG arch: {arch}")

        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def build_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def strip_module(state):
    out = {}
    for k, v in state.items():
        while k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out

def load_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    state = strip_module(state)

    fixed = {}
    for k, v in state.items():
        if k.startswith("classifier1."):
            k = k.replace("classifier1.", "classifier.")
        fixed[k] = v

    return fixed

def build_model(info):
    if info["type"] == "sophon":
        model = build_resnet18()
    elif info["type"] == "vgg":
        model = CustomVGG(info["arch"], NUM_CLASSES)
    else:
        raise ValueError(info["type"])

    state = load_state(info["path"])
    missing, unexpected = model.load_state_dict(state, strict=False)

    return model, missing, unexpected

def apply_bitfit(model):
    for _, p in model.named_parameters():
        p.requires_grad = False

    trainable_names = []

    for name, p in model.named_parameters():
        if name.endswith(".bias") or ".bias" in name:
            p.requires_grad = True
            trainable_names.append(name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    if trainable == 0:
        raise RuntimeError("No bias parameters found for BitFit.")

    return trainable_names, trainable, total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ce = nn.CrossEntropyLoss()

    correct, total, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        out = model(x)
        loss = ce(out, y)

        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item()

    return 100.0 * correct / total, loss_sum / len(loader)

@torch.no_grad()
def save_predictions(model, loader, path):
    model.eval()
    rows = []

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        pred = model(x).argmax(1).cpu().numpy()

        for yt, yp in zip(y.numpy(), pred):
            rows.append([int(yt), int(yp)])

    pd.DataFrame(rows, columns=["y_true", "y_pred"]).to_csv(path, index=False)

def setup_logger(path):
    logger = logging.getLogger(path)
    logger.handlers = []
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(message)s")

    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger

def write_header(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

def save_row(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def train_one(model, train_loader, test_loader, csv_path, logger,
              info, missing, unexpected, trainable_names, trainable, total_params):

    ce = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    trainable_pct = 100.0 * trainable / total_params

    best_acc = -1.0
    best_epoch = -1
    cumulative_time = 0.0

    for epoch in range(1, EPOCHS + 1):
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        epoch_start = time.perf_counter()
        model.train()

        for x, y in tqdm(train_loader, desc=f"{info['name']} bitfit E{epoch}", leave=False):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            optimizer.step()

        acc, loss_val = evaluate(model, test_loader)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - epoch_start
        cumulative_time += epoch_time

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

            pred_file = os.path.join(
                OUT_DIR,
                f"{info['name']}_bitfit_predictions.csv"
            )
            save_predictions(model, test_loader, pred_file)
            logger.info(f"Saved predictions: {pred_file}")

        row = [
            info["method"], info["name"], info["dataset"], info["task"],
            info["arch"], info["path"], "bitfit",
            epoch, EPOCHS, LR, BATCH_SIZE,
            round(acc, 4), round(loss_val, 6),
            round(best_acc, 4), best_epoch,
            round(epoch_time, 4), round(cumulative_time, 4),
            trainable, total_params, round(trainable_pct, 6),
            " | ".join(trainable_names),
            len(missing), len(unexpected),
        ]

        save_row(csv_path, row)

        logger.info(
            f"{info['method']} | {info['name']} | bitfit | "
            f"epoch {epoch:03d}/{EPOCHS} | acc={acc:.2f}% | "
            f"loss={loss_val:.4f} | best={best_acc:.2f}% @ {best_epoch} | "
            f"time={epoch_time:.2f}s | total={cumulative_time:.2f}s"
        )

def write_best_summary(csv_path, summary_path):
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    lines = []
    lines.append("=" * 120)
    lines.append("BEST SUMMARY: BitFit")
    lines.append("=" * 120)
    lines.append("Cell format: BitFit best accuracy (best epoch)")
    lines.append("")

    for (method, ckpt, dataset, task, arch), g in df.groupby(
        ["method", "checkpoint_name", "dataset", "task", "arch"]
    ):
        idx = g["epoch_acc"].idxmax()
        row = g.loc[idx]

        lines.append("")
        lines.append(f"{method} | {ckpt} | {dataset} | {task} | {arch}")
        lines.append("-" * 120)
        lines.append(
            f"BitFit: {row['epoch_acc']:.2f}({int(row['epoch'])}) | "
            f"time={row['cumulative_time_sec']:.2f}s"
        )

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"SAVED SUMMARY: {summary_path}")

def confirm_paths(checkpoints):
    print("\nCHECKING CHECKPOINTS")
    for info in checkpoints:
        if not os.path.exists(info["path"]):
            raise FileNotFoundError(info["path"])
        print("FOUND:", info["path"])
    print("ALL CHECKPOINTS FOUND\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    checkpoints = CHECKPOINTS

    if args.only != "all":
        checkpoints = [c for c in CHECKPOINTS if c["name"] == args.only]

    confirm_paths(checkpoints)

    csv_path = os.path.join(OUT_DIR, "master_bitfit_epoch_results.csv")
    log_path = os.path.join(OUT_DIR, "bitfit.log")
    summary_path = os.path.join(OUT_DIR, "bitfit_best_summary.txt")

    write_header(csv_path)
    logger = setup_logger(log_path)

    logger.info("=" * 100)
    logger.info("STARTING BITFIT")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"LOG: {log_path}")
    logger.info(f"SUMMARY: {summary_path}")
    logger.info("=" * 100)

    loader_cache = {}

    for info in checkpoints:
        try:
            if info["dataset"] not in loader_cache:
                loader_cache[info["dataset"]] = get_loaders(info["dataset"])

            train_loader, test_loader = loader_cache[info["dataset"]]

            logger.info("-" * 100)
            logger.info(f"RUNNING {info['method']} | {info['name']} | BitFit")
            logger.info("-" * 100)

            model, missing, unexpected = build_model(info)
            model = model.to(DEVICE)

            trainable_names, trainable, total_params = apply_bitfit(model)

            logger.info(f"Checkpoint: {info['path']}")
            logger.info(f"Missing keys: {len(missing)}")
            logger.info(f"Unexpected keys: {len(unexpected)}")
            logger.info(f"Trainable bias tensors: {len(trainable_names)}")
            logger.info(f"Trainable bias params: {trainable}")
            logger.info(f"Total params: {total_params}")
            logger.info(f"Trainable percent: {100.0 * trainable / total_params:.6f}%")

            train_one(
                model, train_loader, test_loader,
                csv_path, logger, info,
                missing, unexpected,
                trainable_names, trainable, total_params,
            )

            write_best_summary(csv_path, summary_path)

        except Exception:
            logger.exception(f"FAILED BUT CONTINUING: {info['method']} | {info['name']} | BitFit")
            continue

    write_best_summary(csv_path, summary_path)
    logger.info("DONE BITFIT")
    print("\nALL DONE")

if __name__ == "__main__":
    main()