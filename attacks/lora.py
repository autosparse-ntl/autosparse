
import os, csv, random, logging, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

DATA_ROOT = "./datasets"
OUT_DIR = "./results_clean/lora"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANKS = [4, 8, 16, 32, 64]
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 64
WORKERS = 4
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_CLASSES = 10

LORA_ALPHA_SOPHON = 32
LORA_ALPHA_VGG = 16

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
    "attack","rank","lora_alpha","epoch","epochs","lr","batch_size",
    "epoch_acc","epoch_loss","best_acc_so_far","best_epoch_so_far",
    "epoch_time_sec","cumulative_time_sec",
    "trainable_params","total_params","trainable_percent",
    "num_lora_layers","targeted_layers","missing_keys","unexpected_keys"
]

RE_LABEL = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
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


def load_sophon_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    return strip_module(state)


def load_vgg_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    state = strip_module(state)

    fixed = {}
    for k, v in state.items():
        if k.startswith("classifier1."):
            k = k.replace("classifier1.", "classifier.")
        fixed[k] = v

    return fixed


class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.scale = alpha / rank

        for p in self.linear.parameters():
            p.requires_grad = False

        self.A = nn.Parameter(torch.randn(rank, linear.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(linear.out_features, rank))

    def forward(self, x):
        weight = self.linear.weight + self.scale * (self.B @ self.A)
        return F.linear(x, weight, self.linear.bias)


class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank, alpha):
        super().__init__()
        self.conv = conv
        self.scale = alpha / rank

        for p in self.conv.parameters():
            p.requires_grad = False

        kh, kw = conv.kernel_size
        self.A = nn.Parameter(torch.randn(rank, conv.in_channels * kh * kw) * 0.01)
        self.B = nn.Parameter(torch.zeros(conv.out_channels, rank))

    def forward(self, x):
        delta = self.B @ self.A
        delta = delta.view(
            self.conv.out_channels,
            self.conv.in_channels,
            self.conv.kernel_size[0],
            self.conv.kernel_size[1],
        )

        weight = self.conv.weight + self.scale * delta

        return F.conv2d(
            x,
            weight,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def replace_last_linear(model, rank, alpha):
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    if not names:
        raise RuntimeError("No Linear layer found for simple_lora")

    target = names[-1]
    parent = model
    parts = target.split(".")

    for p in parts[:-1]:
        parent = getattr(parent, p)

    old = getattr(parent, parts[-1])
    setattr(parent, parts[-1], LoRALinear(old, rank, alpha))

    return 1, [target]


def replace_all_lora(module, rank, alpha, prefix=""):
    count = 0
    names = []

    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank, alpha))
            count += 1
            names.append(full)

        elif isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            count += 1
            names.append(full)

        else:
            c, n = replace_all_lora(child, rank, alpha, full)
            count += c
            names.extend(n)

    return count, names


def build_model(info, rank, attack):
    ckpt_path = info["path"]

    if info["type"] == "sophon":
        model = build_resnet18()
        state = load_sophon_state(ckpt_path)
        alpha = LORA_ALPHA_SOPHON

    elif info["type"] == "vgg":
        model = CustomVGG(info["arch"], NUM_CLASSES)
        state = load_vgg_state(ckpt_path)
        alpha = LORA_ALPHA_VGG

    else:
        raise ValueError(info["type"])

    missing, unexpected = model.load_state_dict(state, strict=False)

    freeze_all(model)

    if attack == "simple_lora":
        n_lora, layers = replace_last_linear(model, rank, alpha)
    elif attack == "strong_lora":
        n_lora, layers = replace_all_lora(model, rank, alpha)
    else:
        raise ValueError(attack)

    return model, ckpt_path, missing, unexpected, n_lora, layers, alpha


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


def already_done(csv_path):
    done = set()

    if not os.path.exists(csv_path):
        return done

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for r in reader:
            if int(r["epoch"]) == EPOCHS:
                done.add((r["checkpoint_name"], r["attack"], int(r["rank"])))

    return done


def save_epoch_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def train_one(model, train_loader, test_loader, csv_path, logger, info, attack,
              rank, alpha, ckpt_path, missing, unexpected, n_lora, layers):
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100.0 * trainable / total_params

    best_acc = -1.0
    best_epoch = -1
    cumulative_time = 0.0

    for epoch in range(1, EPOCHS + 1):
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        epoch_start = time.perf_counter()
        model.train()

        for x, y in tqdm(train_loader, desc=f"{info['name']} {attack} R{rank} E{epoch}", leave=False):
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
                f"{info['name']}_{attack}_R{rank}_predictions.csv"
            )
            save_predictions(model, test_loader, pred_file)
            logger.info(f"Saved predictions: {pred_file}")

        row = [
            info["method"], info["name"], info["dataset"], info["task"],
            info["arch"], ckpt_path, attack, rank, alpha,
            epoch, EPOCHS, LR, BATCH_SIZE,
            round(acc, 4), round(loss_val, 6),
            round(best_acc, 4), best_epoch,
            round(epoch_time, 4), round(cumulative_time, 4),
            trainable, total_params, round(trainable_pct, 6),
            n_lora, " | ".join(layers),
            len(missing), len(unexpected),
        ]

        save_epoch_row(csv_path, row)

        logger.info(
            f"{info['method']} | {info['name']} | {attack} | R{rank} | "
            f"epoch {epoch:03d}/{EPOCHS} | acc={acc:.2f}% | "
            f"loss={loss_val:.4f} | best={best_acc:.2f}% @ {best_epoch} | "
            f"time={epoch_time:.2f}s | total={cumulative_time:.2f}s"
        )


def output_files(attack):
    if attack == "simple_lora":
        return (
            os.path.join(OUT_DIR, "master_simple_lora_epoch_results.csv"),
            os.path.join(OUT_DIR, "simple_lora.log"),
        )

    if attack == "strong_lora":
        return (
            os.path.join(OUT_DIR, "master_strong_lora_epoch_results.csv"),
            os.path.join(OUT_DIR, "strong_lora.log"),
        )

    raise ValueError(attack)


def confirm_paths():
    print("\nCHECKING 13 CHECKPOINTS")
    for info in CHECKPOINTS:
        if not os.path.exists(info["path"]):
            raise FileNotFoundError(info["path"])
        print("FOUND:", info["path"])
    print("ALL CHECKPOINTS FOUND\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default="both", choices=["simple_lora", "strong_lora", "both"])
    parser.add_argument("--only", default="all")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    confirm_paths()

    attacks = ["simple_lora", "strong_lora"] if args.attack == "both" else [args.attack]

    loader_cache = {}

    for attack in attacks:
        csv_path, log_path = output_files(attack)
        write_header(csv_path)

        logger = setup_logger(log_path)
        done = already_done(csv_path)

        logger.info("=" * 100)
        logger.info(f"STARTING {attack}")
        logger.info(f"CSV: {csv_path}")
        logger.info(f"LOG: {log_path}")
        logger.info("=" * 100)

        for info in CHECKPOINTS:
            if args.only != "all" and args.only != info["name"]:
                continue

            dataset = info["dataset"]

            if dataset not in loader_cache:
                loader_cache[dataset] = get_loaders(dataset)

            train_loader, test_loader = loader_cache[dataset]

            for rank in RANKS:
                key = (info["name"], attack, rank)

                if key in done:
                    logger.info(f"SKIP existing full run: {key}")
                    continue

                logger.info("-" * 100)
                logger.info(f"RUNNING {info['method']} | {info['name']} | {attack} | R{rank}")
                logger.info("-" * 100)

                model, ckpt_path, missing, unexpected, n_lora, layers, alpha = build_model(
                    info, rank, attack
                )
                model = model.to(DEVICE)

                logger.info(f"Checkpoint: {ckpt_path}")
                logger.info(f"Missing keys: {len(missing)}")
                logger.info(f"Unexpected keys: {len(unexpected)}")
                logger.info(f"LoRA layers: {n_lora}")

                train_one(
                    model, train_loader, test_loader, csv_path, logger,
                    info, attack, rank, alpha, ckpt_path,
                    missing, unexpected, n_lora, layers
                )

        logger.info(f"DONE {attack}")

    print("\nALL DONE")


if __name__ == "__main__":
    main()