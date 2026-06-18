import os, csv, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

ROOT = "."
DATA_ROOT = "./datasets"
CKPT_ROOT = os.path.join(ROOT, "checkpoints")
OUT_DIR = os.path.join(ROOT, "attack", "autosparse_results")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
IMG_SIZE = 64
EPOCHS = 20
BATCH_SIZE = 64
WORKERS = 4

AE_EPOCHS = 4
AE_LR = 1e-3
AE_MAX_BATCHES = 200
SCORE_BATCHES = 200

LR_STAGE1 = 1e-3
WEIGHT_DECAY = 1e-4

SPARSE_RATIO = 0.06
CHANNEL_RATIO = 0.15
MASK_MODE = "channel"
AE_REPAIR_STRENGTH = 8.0
SAMPLE_BETA = 0.50
HEAD_LR_MULT = 5.0
TRAIN_CLASSIFIER = False
TRAIN_FINAL_HEAD_ONLY = True
MODE = "AutoSparse"

AE_RESIDUAL_LAMBDA = 0.02

RE_LABEL = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]

CHECKPOINTS = {
    "sophon_il_cifar10": {"method":"SOPHON-IL","name":"sophon_il_cifar10","type":"sophon","arch":"resnet18","dataset":"CIFAR10","task":"CIFAR10","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_cifar10.pth"),"target_to_beat":88.20},
    "sophon_il_cinic": {"method":"SOPHON-IL","name":"sophon_il_cinic","type":"sophon","arch":"resnet18","dataset":"CINIC","task":"CINIC","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_cinic.pth"),"target_to_beat":76.21},
    "sophon_il_mnist": {"method":"SOPHON-IL","name":"sophon_il_mnist","type":"sophon","arch":"resnet18","dataset":"MNIST","task":"MNIST","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_mnist.pth"),"target_to_beat":99.53},
    "sophon_il_stl10": {"method":"SOPHON-IL","name":"sophon_il_stl10","type":"sophon","arch":"resnet18","dataset":"STL10","task":"STL10","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_stl10.pth"),"target_to_beat":78.64},
    "sophon_il_svhn": {"method":"SOPHON-IL","name":"sophon_il_svhn","type":"sophon","arch":"resnet18","dataset":"SVHN","task":"SVHN","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_svhn.pth"),"target_to_beat":94.94},

    "sophon_kl_cifar10": {"method":"SOPHON-KL","name":"sophon_kl_cifar10","type":"sophon","arch":"resnet18","dataset":"CIFAR10","task":"CIFAR10","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_cifar10.pth"),"target_to_beat":88.59},
    "sophon_kl_cinic": {"method":"SOPHON-KL","name":"sophon_kl_cinic","type":"sophon","arch":"resnet18","dataset":"CINIC","task":"CINIC","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_cinic.pth"),"target_to_beat":76.05},
    "sophon_kl_mnist": {"method":"SOPHON-KL","name":"sophon_kl_mnist","type":"sophon","arch":"resnet18","dataset":"MNIST","task":"MNIST","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_mnist.pth"),"target_to_beat":99.51},
    "sophon_kl_stl10": {"method":"SOPHON-KL","name":"sophon_kl_stl10","type":"sophon","arch":"resnet18","dataset":"STL10","task":"STL10","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_stl10.pth"),"target_to_beat":78.40},
    "sophon_kl_svhn": {"method":"SOPHON-KL","name":"sophon_kl_svhn","type":"sophon","arch":"resnet18","dataset":"SVHN","task":"SVHN","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_svhn.pth"),"target_to_beat":95.00},

    "hntl_cifar_stl_vgg13": {"method":"HNTL","name":"hntl_cifar_stl_vgg13","type":"vgg","arch":"vgg13","dataset":"STL10","task":"CIFAR10_to_STL10","path":os.path.join(CKPT_ROOT,"hntl_cifar_stl_vgg13.pth"),"target_to_beat":87.00},
    "cupi_cifar_stl_vgg13": {"method":"CUPI","name":"cupi_cifar_stl_vgg13","type":"vgg","arch":"vgg13","dataset":"STL10","task":"CIFAR10_to_STL10","path":os.path.join(CKPT_ROOT,"cupi_cifar_stl_vgg13.pth"),"target_to_beat":85.45},
    "cuti_cifar_stl_vgg19": {"method":"CUTI","name":"cuti_cifar_stl_vgg19","type":"vgg","arch":"vgg19","dataset":"STL10","task":"CIFAR10_to_STL10","path":os.path.join(CKPT_ROOT,"cuti_cifar_stl_vgg19.pth"),"target_to_beat":77.06},
}

CSV_PATH = os.path.join(OUT_DIR, "master_autosparse.csv")
SUMMARY_PATH = None
SEED_SUMMARY_CSV = os.path.join(OUT_DIR, "autosparse_seed_summary.csv")

CSV_HEADER = [
    "seed","method","checkpoint_name","dataset","task","arch","checkpoint",
    "mode","epoch","epochs","epoch_acc","epoch_loss","best_acc_so_far","best_epoch_so_far",
    "target_to_beat","beat_target","epoch_time_sec","cumulative_time_sec",
    "effective_trainable","total_params","effective_percent",
    "mask_mode","channel_ratio","sparse_ratio","repair_strength","sample_beta","head_lr_mult",
    "lr_stage1","ae_epochs","ae_max_batches","score_batches","ae_residual_lambda",
    "num_conv_layers_scored","num_conv_layers_trained","missing_keys","unexpected_keys"
]

CURRENT_SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def write_header():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

def save_row(row):
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)

def save_seed_summary(row):
    header = [
        "seed", "method", "checkpoint_name", "dataset", "task", "arch",
        "best_acc", "best_epoch", "target_to_beat", "beat_target",
        "effective_trainable", "total_params", "effective_percent",
        "channel_ratio", "sparse_ratio", "repair_strength",
        "sample_beta", "head_lr_mult", "ae_epochs",
        "ae_max_batches", "score_batches", "ae_residual_lambda"
    ]

    exists = os.path.exists(SEED_SUMMARY_CSV)

    with open(SEED_SUMMARY_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)

def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

class STL10Mapped(datasets.STL10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, RE_LABEL[target]

def get_loaders(dataset):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    tf_rgb_train = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    tf_rgb_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    tf_mnist = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize,
    ])

    d = dataset.upper()

    if d == "CIFAR10":
        train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=tf_rgb_train)
        test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=tf_rgb_test)

    elif d == "MNIST":
        train_set = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tf_mnist)
        test_set = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=tf_mnist)

    elif d == "SVHN":
        train_set = datasets.SVHN(DATA_ROOT, split="train", download=False, transform=tf_rgb_train)
        test_set = datasets.SVHN(DATA_ROOT, split="test", download=False, transform=tf_rgb_test)

    elif d == "STL10":
        train_set = STL10Mapped(DATA_ROOT, split="train", download=False, transform=tf_rgb_train)
        test_set = STL10Mapped(DATA_ROOT, split="test", download=False, transform=tf_rgb_test)

    elif d == "CINIC":
        cinic_root = os.path.join(DATA_ROOT, "CINIC10")
        train_dir = os.path.join(cinic_root, "train")
        valid_dir = os.path.join(cinic_root, "valid")
        test_dir = os.path.join(cinic_root, "test")

        if os.path.exists(train_dir) and os.path.exists(valid_dir):
            train_set = datasets.ImageFolder(train_dir, transform=tf_rgb_train)
            test_set = datasets.ImageFolder(valid_dir, transform=tf_rgb_test)
        elif os.path.exists(train_dir) and os.path.exists(test_dir):
            train_set = datasets.ImageFolder(train_dir, transform=tf_rgb_train)
            test_set = datasets.ImageFolder(test_dir, transform=tf_rgb_test)
        else:
            raise FileNotFoundError(f"CINIC folders not found inside {cinic_root}")

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    g = torch.Generator()
    g.manual_seed(CURRENT_SEED)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=(DEVICE == "cuda"),
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

def build_model(info):
    if info["type"] == "sophon":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    elif info["type"] == "vgg":
        model = CustomVGG(info["arch"], NUM_CLASSES)

    else:
        raise ValueError(info["type"])

    state = load_state(info["path"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    return model, missing, unexpected

def safe_name(name):
    return name.replace(".", "__")

def get_sample_beta(epoch, total_epochs, max_beta):
    warmup = max(1, total_epochs // 4)
    if epoch <= warmup:
        return 0.0
    return max_beta * min(1.0, (epoch - warmup) / max(1, total_epochs - warmup))

class GenericConvHook:
    def __init__(self, model):
        self.outputs = {}
        self.handles = []
        self.layer_names = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.layer_names.append(name)
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inp, out):
            if out.ndim == 4:
                pooled = F.adaptive_avg_pool2d(out.detach(), 1).flatten(1)
                self.outputs[name] = pooled
        return hook

    def clear(self):
        self.outputs = {}

    def remove(self):
        for h in self.handles:
            h.remove()

class TrainConvHook:
    def __init__(self, model):
        self.outputs = {}
        self.handles = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inp, out):
            if out.ndim == 4:
                pooled = F.adaptive_avg_pool2d(out, 1).flatten(1)
                self.outputs[name] = pooled
        return hook

    def clear(self):
        self.outputs = {}

    def remove(self):
        for h in self.handles:
            h.remove()

class AEProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = max(32, dim // 2)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)

def build_ae_bank(model, train_loader):
    hooker = GenericConvHook(model)
    model.eval()

    x, _ = next(iter(train_loader))
    x = x.to(DEVICE)

    with torch.no_grad():
        hooker.clear()
        _ = model(x)

    ae_bank = nn.ModuleDict()
    layer_dims = {}

    for name, feat in hooker.outputs.items():
        dim = feat.shape[1]
        ae = AEProbe(dim).to(DEVICE)

        torch.manual_seed(CURRENT_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CURRENT_SEED)

        for p in ae.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
            else:
                nn.init.zeros_(p)

        ae_bank[safe_name(name)] = ae
        layer_dims[name] = dim

    hooker.remove()
    return ae_bank, layer_dims

def train_ae_damage(model, train_loader):
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    ae_bank, layer_dims = build_ae_bank(model, train_loader)
    hooker = GenericConvHook(model)

    score_loader = DataLoader(
        train_loader.dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        ae_bank.parameters(),
        lr=AE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    print("=" * 100)
    print("AE SUPPRESSED-CHANNEL DISCOVERY")
    print(f"Conv layers detected: {len(layer_dims)}")
    print(f"AE_EPOCHS={AE_EPOCHS} | AE_MAX_BATCHES={AE_MAX_BATCHES}")
    print("=" * 100)

    for ep in range(1, AE_EPOCHS + 1):
        ae_bank.train()
        total_loss = 0.0
        count = 0
        start = time.perf_counter()

        for i, (x, _) in enumerate(tqdm(train_loader, desc=f"AE damage E{ep}", leave=False)):
            if i >= AE_MAX_BATCHES:
                break

            x = x.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                hooker.clear()
                _ = model(x)
                feats = {k: v.detach() for k, v in hooker.outputs.items()}

            loss = 0.0
            used = 0

            for name, feat in feats.items():
                key = safe_name(name)
                if key not in ae_bank:
                    continue

                rec = ae_bank[key](feat)
                loss = loss + F.mse_loss(rec, feat)
                used += 1

            if used == 0:
                continue

            loss = loss / used
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            count += 1

        print(
            f"AE | epoch {ep:02d}/{AE_EPOCHS} | "
            f"loss={total_loss / max(count, 1):.6f} | "
            f"time={time.perf_counter() - start:.1f}s"
        )

    scores = compute_channel_scores(model, ae_bank, score_loader, hooker)
    hooker.remove()
    return ae_bank, scores, len(layer_dims)

@torch.no_grad()
def compute_channel_scores(model, ae_bank, train_loader, hooker):
    model.eval()
    ae_bank.eval()

    err_sum = {}
    count = 0

    for i, (x, _) in enumerate(tqdm(train_loader, desc="AE channel scores", leave=False)):
        if i >= SCORE_BATCHES:
            break

        x = x.to(DEVICE, non_blocking=True)

        hooker.clear()
        _ = model(x)

        for name, feat in hooker.outputs.items():
            key = safe_name(name)
            if key not in ae_bank:
                continue

            rec = ae_bank[key](feat)
            err = (feat - rec).pow(2).mean(dim=0)

            if name not in err_sum:
                err_sum[name] = err.detach().cpu()
            else:
                err_sum[name] += err.detach().cpu()

        count += 1

    scores = {}
    for name, err in err_sum.items():
        err = err / max(count, 1)
        err = err / (err.max() + 1e-8)
        scores[name] = err.float()

    print("=" * 100)
    print("AE-ONLY CHANNEL SCORE SUMMARY")
    for name, score in scores.items():
        print(f"{name:35s} | channels={score.numel():4d} | mean_score={score.mean().item():.6f}")
    print("=" * 100)

    return scores

def make_ae_mask(layer_name, weight, channel_score, ratio):
    with torch.no_grad():
        if weight.ndim != 4:
            flat = weight.detach().abs().flatten()
            k = max(1, int(flat.numel() * ratio))
            th = torch.topk(flat, k=k, largest=True).values[-1]
            return (weight.detach().abs() >= th).float()

        out_c = weight.shape[0]

        if channel_score is None:
            score = torch.ones(out_c, device=weight.device)
        else:
            score = channel_score[:out_c].to(weight.device)
            if score.numel() < out_c:
                pad = torch.ones(out_c - score.numel(), device=weight.device) * score.mean().clamp_min(1e-8)
                score = torch.cat([score, pad], dim=0)

        if MASK_MODE == "channel":
            k = max(1, int(out_c * CHANNEL_RATIO))
            idx = torch.topk(score, k=k, largest=True).indices
            mask = torch.zeros_like(weight)
            mask[idx, :, :, :] = 1.0
            return mask

        score = score.view(out_c, 1, 1, 1)
        priority = weight.detach().abs() * (1.0 + AE_REPAIR_STRENGTH * score.pow(2))
        flat = priority.flatten()
        k = max(1, int(flat.numel() * ratio))
        th = torch.topk(flat, k=k, largest=True).values[-1]
        return (priority >= th).float()

def apply_aecr_masks(model, channel_scores):
    for p in model.parameters():
        p.requires_grad = False

    masks = {}
    originals = {}
    effective = 0
    trained_conv_layers = 0

    linear_layers = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, nn.Linear)
    ]
    final_linear_name = linear_layers[-1][0] if linear_layers else None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            score = channel_scores.get(name, None)
            mask = make_ae_mask(name, module.weight.data, score, SPARSE_RATIO)

            module.weight.requires_grad = True
            effective += int(mask.sum().item())
            trained_conv_layers += 1

            def hook_fn(grad, m=mask):
                return grad * m.to(grad.device)

            module.weight.register_hook(hook_fn)
            masks[module.weight] = mask.to(module.weight.device)
            originals[module.weight] = module.weight.detach().clone()

            if module.bias is not None:
                module.bias.requires_grad = True
                bmask = torch.ones_like(module.bias.data)
                masks[module.bias] = bmask
                originals[module.bias] = module.bias.detach().clone()
                effective += module.bias.numel()

        if isinstance(module, nn.Linear):
            if TRAIN_FINAL_HEAD_ONLY and name == final_linear_name:
                module.weight.requires_grad = True
                module.bias.requires_grad = True
                effective += module.weight.numel() + module.bias.numel()
            elif TRAIN_CLASSIFIER:
                module.weight.requires_grad = True
                module.bias.requires_grad = True
                effective += module.weight.numel() + module.bias.numel()

    return masks, originals, effective, trained_conv_layers

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

@torch.no_grad()
def save_predictions(model, loader, path):
    model.eval()
    rows = []

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        pred = model(x).argmax(1).cpu().numpy()

        for yt, yp in zip(y.numpy(), pred):
            rows.append([int(yt), int(yp)])

    import pandas as pd
    pd.DataFrame(rows, columns=["y_true", "y_pred"]).to_csv(path, index=False)

def train_recovery(info, model, missing, unexpected, channel_scores, num_scored_layers, train_loader, test_loader, ae_bank=None):
    masks, originals, effective, trained_conv_layers = apply_aecr_masks(model, channel_scores)
    model = model.to(DEVICE)

    if ae_bank is not None:
        ae_bank = ae_bank.to(DEVICE)
        ae_bank.eval()
        for p in ae_bank.parameters():
            p.requires_grad = False

    train_hook = TrainConvHook(model)

    total = count_total_params(model)
    pct = 100.0 * effective / total

    head_params = []
    backbone_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".fc." in name or name.startswith("fc.") or "classifier" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_STAGE1, "group_name": "backbone"},
            {"params": head_params, "lr": LR_STAGE1 * HEAD_LR_MULT, "group_name": "head"},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR_STAGE1 * 0.05,
    )

    best_acc = -1.0
    best_epoch = -1
    cumulative = 0.0
    target = info["target_to_beat"]

    print("=" * 100)
    print("AutoSparse RECOVERY")
    print(f"Checkpoint: {info['name']}")
    print(f"Arch      : {info['arch']}")
    print(f"Dataset   : {info['dataset']}")
    print(f"Target    : {target:.2f}%")
    print(f"Effective : {effective:,}/{total:,} ({pct:.4f}%)")
    print(f"Conv scored/trained: {num_scored_layers}/{trained_conv_layers}")
    print(f"Mask={MASK_MODE} | Channel ratio={CHANNEL_RATIO} | Sparse ratio={SPARSE_RATIO}")
    print(f"LR={LR_STAGE1} | Head LR mult={HEAD_LR_MULT} | Cosine annealing T_max={EPOCHS}")
    print("=" * 100)

    for epoch in range(1, EPOCHS + 1):
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        model.train()

        for x, y in tqdm(train_loader, desc=f"{info['name']} AutoSparse E{epoch}", leave=False):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            train_hook.clear()
            logits = model(x)

            ae_loss = 0.0
            used = 0

            if ae_bank is not None:
                for name, feat in train_hook.outputs.items():
                    key = safe_name(name)
                    if key in ae_bank:
                        rec = ae_bank[key](feat)
                        ae_loss = ae_loss + F.mse_loss(rec, feat)
                        used += 1

                if used > 0:
                    ae_loss = ae_loss / used

            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                conf = probs.max(dim=1).values
                current_beta = get_sample_beta(epoch, EPOCHS, SAMPLE_BETA)
                sw = 1.0 + current_beta * (1.0 - conf)
                sw = sw / sw.mean().clamp_min(1e-8)

            loss_each = F.cross_entropy(logits, y, reduction="none")
            loss = (loss_each * sw).mean()

            if used > 0 and AE_RESIDUAL_LAMBDA > 0.0:
                loss = loss + AE_RESIDUAL_LAMBDA * ae_loss

            loss.backward()
            optimizer.step()
            restore_unselected(masks, originals)

        scheduler.step()

        acc, loss_val = evaluate(model, test_loader)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        cumulative += elapsed

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

            pred_file = os.path.join(
                OUT_DIR,
                f"{info['name']}_autosparse_seed{CURRENT_SEED}_predictions.csv"
            )

            save_predictions(model, test_loader, pred_file)
            print(f"Saved predictions: {pred_file}")

        beat = best_acc > target

        save_row([
            CURRENT_SEED,
            info["method"], info["name"], info["dataset"], info["task"], info["arch"], info["path"],
            MODE, epoch, EPOCHS,
            round(acc, 4), round(loss_val, 6),
            round(best_acc, 4), best_epoch,
            target, beat,
            round(elapsed, 4), round(cumulative, 4),
            effective, total, round(pct, 6),
            MASK_MODE, CHANNEL_RATIO, SPARSE_RATIO, AE_REPAIR_STRENGTH, SAMPLE_BETA, HEAD_LR_MULT,
            LR_STAGE1,
            AE_EPOCHS, AE_MAX_BATCHES, SCORE_BATCHES, AE_RESIDUAL_LAMBDA,
            num_scored_layers, trained_conv_layers,
            len(missing), len(unexpected),
        ])

        current_lr = optimizer.param_groups[0]["lr"]
        current_head_lr = optimizer.param_groups[1]["lr"]

        print(
            f"AutoSparse | epoch {epoch:03d}/{EPOCHS} | "
            f"acc={acc:.2f}% | loss={loss_val:.4f} | "
            f"best={best_acc:.2f}% @ {best_epoch} | "
            f"target={target:.2f}% | beat={beat} | "
            f"bb_lr={current_lr:.6f} | head_lr={current_head_lr:.6f} | "
            f"time={elapsed:.1f}s"
        )

    with open(SUMMARY_PATH, "a") as f:
        f.write(
            f"\n{MODE} | seed={CURRENT_SEED} | {info['method']} | {info['name']} | "
            f"best={best_acc:.2f}({best_epoch}) | target={target:.2f} | beat={best_acc > target} | "
            f"effective={effective} | pct={pct:.4f}% | mask={MASK_MODE} | "
            f"channel_ratio={CHANNEL_RATIO} | sparse_ratio={SPARSE_RATIO}\n"
        )

    save_seed_summary([
        CURRENT_SEED,
        info["method"],
        info["name"],
        info["dataset"],
        info["task"],
        info["arch"],
        round(best_acc, 4),
        best_epoch,
        target,
        best_acc > target,
        effective,
        total,
        round(pct, 6),
        CHANNEL_RATIO,
        SPARSE_RATIO,
        AE_REPAIR_STRENGTH,
        SAMPLE_BETA,
        HEAD_LR_MULT,
        AE_EPOCHS,
        AE_MAX_BATCHES,
        SCORE_BATCHES,
        AE_RESIDUAL_LAMBDA,
    ])

    print("=" * 100)
    print("FINAL SUMMARY")
    print(f"Checkpoint : {info['name']}")
    print(f"Best       : {best_acc:.2f}% @ epoch {best_epoch}")
    print(f"Target     : {target:.2f}%")
    print(f"Beat       : {best_acc > target}")
    print(f"Effective  : {effective:,}/{total:,} ({pct:.4f}%)")
    print(f"CSV        : {CSV_PATH}")
    print("=" * 100)

    train_hook.remove()

def run_one(info):
    if not os.path.exists(info["path"]):
        print(f"SKIPPING missing checkpoint: {info['path']}")
        return

    train_loader, test_loader = get_loaders(info["dataset"])

    model, missing, unexpected = build_model(info)
    model = model.to(DEVICE)

    ae_bank, channel_scores, num_scored_layers = train_ae_damage(model, train_loader)

    model, missing, unexpected = build_model(info)
    model = model.to(DEVICE)

    train_recovery(
        info,
        model,
        missing,
        unexpected,
        channel_scores,
        num_scored_layers,
        train_loader,
        test_loader,
        ae_bank,
    )

def confirm_paths(selected):
    print("=" * 100)
    print("CHECKING CHECKPOINTS")
    print("=" * 100)

    for key, info in selected.items():
        flag = "FOUND" if os.path.exists(info["path"]) else "MISSING"
        print(f"{flag}: {key} -> {info['path']}")

    print("=" * 100)

def main():
    global CURRENT_SEED, EPOCHS, BATCH_SIZE, SPARSE_RATIO, CHANNEL_RATIO, MASK_MODE
    global AE_EPOCHS, AE_REPAIR_STRENGTH, SAMPLE_BETA, HEAD_LR_MULT
    global LR_STAGE1, AE_MAX_BATCHES, SCORE_BATCHES, AE_RESIDUAL_LAMBDA, SUMMARY_PATH

    parser = argparse.ArgumentParser()

    parser.add_argument("--only", default="pilot", choices=["pilot", "all"] + list(CHECKPOINTS.keys()))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)

    parser.add_argument("--sparse_ratio", type=float, default=0.06)
    parser.add_argument("--channel_ratio", type=float, default=0.15)
    parser.add_argument("--mask_mode", default="channel", choices=["weight", "channel"])

    parser.add_argument("--repair_strength", type=float, default=8.0)
    parser.add_argument("--sample_beta", type=float, default=0.50)
    parser.add_argument("--head_lr_mult", type=float, default=5.0)

    parser.add_argument("--lr_stage1", type=float, default=1e-3)

    parser.add_argument("--ae_epochs", type=int, default=4)
    parser.add_argument("--ae_max_batches", type=int, default=200)
    parser.add_argument("--score_batches", type=int, default=200)
    parser.add_argument("--ae_residual_lambda", type=float, default=0.02)

    args = parser.parse_args()

    CURRENT_SEED = args.seed

    SUMMARY_PATH = os.path.join(
        OUT_DIR,
        f"autosparse_summary_seed_{CURRENT_SEED}.txt"
    )

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    SPARSE_RATIO = args.sparse_ratio
    CHANNEL_RATIO = args.channel_ratio
    MASK_MODE = args.mask_mode

    AE_REPAIR_STRENGTH = args.repair_strength
    SAMPLE_BETA = args.sample_beta
    HEAD_LR_MULT = args.head_lr_mult

    LR_STAGE1 = args.lr_stage1

    AE_EPOCHS = args.ae_epochs
    AE_MAX_BATCHES = args.ae_max_batches
    SCORE_BATCHES = args.score_batches
    AE_RESIDUAL_LAMBDA = args.ae_residual_lambda

    set_seed(CURRENT_SEED)
    write_header()

    if args.only == "pilot":
        selected = {
            "sophon_il_cifar10": CHECKPOINTS["sophon_il_cifar10"],
            "hntl_cifar_stl_vgg13": CHECKPOINTS["hntl_cifar_stl_vgg13"],
            "cupi_cifar_stl_vgg13": CHECKPOINTS["cupi_cifar_stl_vgg13"],
        }
    elif args.only == "all":
        selected = CHECKPOINTS
    else:
        selected = {args.only: CHECKPOINTS[args.only]}

    confirm_paths(selected)

    for _, info in selected.items():
        run_one(info)

if __name__ == "__main__":
    main()