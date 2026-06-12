import os, csv, time, random, argparse, logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

ROOT = "."
DATA_ROOT = "./datasets"
CKPT_ROOT = os.path.join(ROOT, "checkpoints")
OUT_DIR = os.path.join(ROOT, "attack", "sparse_ft_1_5_10_results")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPARSITIES = [0.01, 0.05, 0.10]
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 64
WORKERS = 4
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_CLASSES = 10

RE_LABEL = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]

CHECKPOINTS = [
    {"method":"SOPHON-IL","name":"sophon_il_cifar10","type":"sophon","arch":"resnet18","dataset":"CIFAR10","task":"CIFAR10","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_cifar10.pth")},
    {"method":"SOPHON-IL","name":"sophon_il_cinic","type":"sophon","arch":"resnet18","dataset":"CINIC","task":"CINIC","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_cinic.pth")},
    {"method":"SOPHON-IL","name":"sophon_il_mnist","type":"sophon","arch":"resnet18","dataset":"MNIST","task":"MNIST","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_mnist.pth")},
    {"method":"SOPHON-IL","name":"sophon_il_stl10","type":"sophon","arch":"resnet18","dataset":"STL10","task":"STL10","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_stl10.pth")},
    {"method":"SOPHON-IL","name":"sophon_il_svhn","type":"sophon","arch":"resnet18","dataset":"SVHN","task":"SVHN","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_svhn.pth")},

    {"method":"SOPHON-KL","name":"sophon_kl_cifar10","type":"sophon","arch":"resnet18","dataset":"CIFAR10","task":"CIFAR10","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_cifar10.pth")},
    {"method":"SOPHON-KL","name":"sophon_kl_cinic","type":"sophon","arch":"resnet18","dataset":"CINIC","task":"CINIC","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_cinic.pth")},
    {"method":"SOPHON-KL","name":"sophon_kl_mnist","type":"sophon","arch":"resnet18","dataset":"MNIST","task":"MNIST","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_mnist.pth")},
    {"method":"SOPHON-KL","name":"sophon_kl_stl10","type":"sophon","arch":"resnet18","dataset":"STL10","task":"STL10","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_stl10.pth")},
    {"method":"SOPHON-KL","name":"sophon_kl_svhn","type":"sophon","arch":"resnet18","dataset":"SVHN","task":"SVHN","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_svhn.pth")},

    {"method":"HNTL","name":"hntl_cifar_stl_vgg13","type":"vgg","arch":"vgg13","dataset":"STL10","task":"CIFAR10_to_STL10","path":os.path.join(CKPT_ROOT,"hntl_cifar_stl_vgg13.pth")},
    {"method":"CUPI","name":"cupi_cifar_stl_vgg13","type":"vgg","arch":"vgg13","dataset":"STL10","task":"CIFAR10_to_STL10","path":os.path.join(CKPT_ROOT,"cupi_cifar_stl_vgg13.pth")},
    {"method":"CUTI","name":"cuti_cifar_stl_vgg19","type":"vgg","arch":"vgg19","dataset":"STL10","task":"CIFAR10_to_STL10","path":os.path.join(CKPT_ROOT,"cuti_cifar_stl_vgg19.pth")},
]

CSV_HEADER = [
    "seed","method","checkpoint_name","dataset","task","arch","checkpoint",
    "attack","sparsity","epoch","epochs","lr","batch_size",
    "epoch_acc","epoch_loss","best_acc_so_far","best_epoch_so_far",
    "epoch_time_sec","cumulative_time_sec",
    "trainable_params","total_params","trainable_percent",
    "targeted_layers","missing_keys","unexpected_keys"
]

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
        cinic_root = os.path.join(DATA_ROOT, "CINIC10")
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

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def make_sparse_mask(param, sparsity):
    flat = param.detach().abs().flatten()
    k = max(1, int(flat.numel() * sparsity))
    threshold = torch.topk(flat, k=k, largest=True).values[-1]
    return (param.detach().abs() >= threshold).float()

def apply_sparseft(model, sparsity, attack):
    freeze_all(model)

    masks = {}
    originals = {}
    trainable_params = 0
    targeted_layers = []

    linear_layers = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, nn.Linear)
    ]
    final_linear_name = linear_layers[-1][0] if linear_layers else None

    for name, module in model.named_modules():
        use_layer = False

        if attack == "simple_sparseft":
            if isinstance(module, nn.Linear) and name == final_linear_name:
                use_layer = True

        elif attack == "strong_sparseft":
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                use_layer = True

        else:
            raise ValueError(attack)

        if not use_layer:
            continue

        if hasattr(module, "weight") and module.weight is not None:
            mask = make_sparse_mask(module.weight.data, sparsity).to(module.weight.device)
            module.weight.requires_grad = True

            def hook_fn(grad, m=mask):
                return grad * m.to(grad.device)

            module.weight.register_hook(hook_fn)

            masks[module.weight] = mask
            originals[module.weight] = module.weight.detach().clone()

            trainable_params += int(mask.sum().item())
            targeted_layers.append(name + ".weight")

        if hasattr(module, "bias") and module.bias is not None:
            module.bias.requires_grad = True

            bmask = torch.ones_like(module.bias.data).to(module.bias.device)

            masks[module.bias] = bmask
            originals[module.bias] = module.bias.detach().clone()

            trainable_params += module.bias.numel()
            targeted_layers.append(name + ".bias")

    return masks, originals, trainable_params, targeted_layers

def restore_unselected(masks, originals):
    with torch.no_grad():
        for p, m in masks.items():
            p.data.copy_(p.data * m + originals[p] * (1.0 - m))

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ce = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        out = model(x)
        loss = ce(out, y)

        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item()

    return 100.0 * correct / total, loss_sum / len(loader)

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
                done.add((
                    r["checkpoint_name"],
                    r["attack"],
                    float(r["sparsity"]),
                ))

    return done

def save_epoch_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def train_one(
    model,
    train_loader,
    test_loader,
    csv_path,
    logger,
    info,
    attack,
    sparsity,
    missing,
    unexpected,
):
    masks, originals, trainable, layers = apply_sparseft(model, sparsity, attack)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    ce = nn.CrossEntropyLoss()

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

        for x, y in tqdm(
            train_loader,
            desc=f"{info['name']} {attack} S{sparsity} E{epoch}",
            leave=False,
        ):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = ce(logits, y)

            loss.backward()
            optimizer.step()

            restore_unselected(masks, originals)

        acc, loss_val = evaluate(model, test_loader)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - epoch_start
        cumulative_time += epoch_time

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        row = [
            SEED,
            info["method"],
            info["name"],
            info["dataset"],
            info["task"],
            info["arch"],
            info["path"],
            attack,
            sparsity,
            epoch,
            EPOCHS,
            LR,
            BATCH_SIZE,
            round(acc, 4),
            round(loss_val, 6),
            round(best_acc, 4),
            best_epoch,
            round(epoch_time, 4),
            round(cumulative_time, 4),
            trainable,
            total_params,
            round(trainable_pct, 6),
            " | ".join(layers),
            len(missing),
            len(unexpected),
        ]

        save_epoch_row(csv_path, row)

        logger.info(
            f"{info['method']} | {info['name']} | {attack} | "
            f"sparsity={sparsity:.2f} | epoch {epoch:03d}/{EPOCHS} | "
            f"acc={acc:.2f}% | loss={loss_val:.4f} | "
            f"best={best_acc:.2f}% @ {best_epoch} | "
            f"trainable={trainable_pct:.4f}% | "
            f"time={epoch_time:.2f}s | total={cumulative_time:.2f}s"
        )

def output_files(attack):
    if attack == "simple_sparseft":
        return (
            os.path.join(
                OUT_DIR,
                "master_simple_sparseft_1_5_10_epoch_results.csv",
            ),
            os.path.join(OUT_DIR, "simple_sparseft_1_5_10.log"),
        )

    if attack == "strong_sparseft":
        return (
            os.path.join(
                OUT_DIR,
                "master_strong_sparseft_1_5_10_epoch_results.csv",
            ),
            os.path.join(OUT_DIR, "strong_sparseft_1_5_10.log"),
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

    parser.add_argument(
        "--attack",
        default="both",
        choices=["simple_sparseft", "strong_sparseft", "both"],
    )
    parser.add_argument("--only", default="all")

    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    confirm_paths()

    attacks = (
        ["simple_sparseft", "strong_sparseft"]
        if args.attack == "both"
        else [args.attack]
    )

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

            for sparsity in SPARSITIES:
                key = (info["name"], attack, float(sparsity))

                if key in done:
                    logger.info(f"SKIP existing full run: {key}")
                    continue

                logger.info("-" * 100)
                logger.info(
                    f"RUNNING {info['method']} | {info['name']} | "
                    f"{attack} | sparsity={sparsity}"
                )
                logger.info("-" * 100)

                model, missing, unexpected = build_model(info)

                train_one(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    csv_path=csv_path,
                    logger=logger,
                    info=info,
                    attack=attack,
                    sparsity=sparsity,
                    missing=missing,
                    unexpected=unexpected,
                )

        logger.info(f"DONE {attack}")

    print("\nALL DONE")
    print(f"Output folder: {OUT_DIR}")

if __name__ == "__main__":
    main()