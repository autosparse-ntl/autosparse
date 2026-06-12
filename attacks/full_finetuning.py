import os
import csv
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        while k.startswith("module."):
            k = k[7:]
        k = k.replace("classifier1.", "classifier.")
        new_state[k] = v
    return new_state


def build_model(arch, num_classes=10):
    if arch == "vgg13":
        model = models.vgg13(weights=None)
    elif arch == "vgg19":
        model = models.vgg19(weights=None)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    # NTLBench VGG uses 64x64 input -> final feature size 512*2*2 = 2048
    model.avgpool = nn.Identity()
    model.classifier = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(256, 256),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(256, num_classes),
    )
    return model


def get_stl10_loaders(data_root, batch_size, num_workers, args):
    tfm_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    tfm_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = datasets.STL10(
        root=data_root,
        split="train",
        download=True,
        transform=tfm_train,
    )

    n = int(len(train_set) * getattr(args, "data_frac", 0.1))
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(train_set), generator=g)[:n]
    train_set = torch.utils.data.Subset(train_set, idx)


    test_set = datasets.STL10(
        root=data_root,
        split="test",
        download=True,
        transform=tfm_test,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = criterion(out, y)

        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)

    acc = 100.0 * correct / total
    avg_loss = loss_sum / total
    return acc, avg_loss


def train_ft(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)

    train_loader, test_loader = get_stl10_loaders(
        args.data_root,
        args.batch_size,
        args.num_workers,
        args,
    )

    model = build_model(args.arch, num_classes=10)

    state = torch.load(args.ckpt, map_location="cpu")
    state = clean_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Loaded checkpoint:", args.ckpt)
    print("Arch:", args.arch)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model = model.to(device)

    init_acc, init_loss = evaluate(model, test_loader, device)
    print(f"[Before FT] target_acc={init_acc:.2f}, target_loss={init_loss:.4f}")

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    with open(args.save_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "arch",
            "checkpoint",
            "epoch",
            "train_loss",
            "target_acc",
            "target_loss",
            "optimizer",
            "lr",
            "batch_size",
            "epochs",
        ])

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total = 0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * y.size(0)
                total += y.size(0)

            train_loss = total_loss / total
            target_acc, target_loss = evaluate(model, test_loader, device)

            writer.writerow([
                args.method,
                args.arch,
                args.ckpt,
                epoch,
                train_loss,
                target_acc,
                target_loss,
                args.optimizer,
                args.lr,
                args.batch_size,
                args.epochs,
            ])

            print(
                f"[FT] epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"target_acc={target_acc:.2f}"
            )

    print("Saved CSV:", args.save_csv)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--arch", type=str, required=True, choices=["vgg13", "vgg19"])

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_csv", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--data_frac", type=float, default=0.1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ft(args)
