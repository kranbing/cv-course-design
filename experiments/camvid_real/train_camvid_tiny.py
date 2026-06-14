import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


VOID_CLASS = 30


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class CamVidTiny(Dataset):
    def __init__(self, root: Path, split: str, image_size=(96, 128), train_ratio=0.8):
        self.root = Path(root)
        self.image_size = image_size
        images = sorted((self.root / "images").glob("*.png"))
        pairs = []
        for img in images:
            lab = self.root / "labels" / f"{img.stem}_P.png"
            if lab.exists():
                pairs.append((img, lab))
        rng = random.Random(42)
        rng.shuffle(pairs)
        cut = int(len(pairs) * train_ratio)
        self.pairs = pairs[:cut] if split == "train" else pairs[cut:]
        self.split = split

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lab_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(lab_path)
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        label = TF.resize(label, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        if self.split == "train" and random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        label = torch.from_numpy(np.array(label, dtype=np.int64))
        return image, label, img_path.name


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UNetSmall(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.e1 = nn.Sequential(ConvBNReLU(3, 32), ConvBNReLU(32, 32))
        self.e2 = nn.Sequential(ConvBNReLU(32, 64), ConvBNReLU(64, 64))
        self.e3 = nn.Sequential(ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = nn.Sequential(ConvBNReLU(128, 64), ConvBNReLU(64, 64))
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = nn.Sequential(ConvBNReLU(64, 32), ConvBNReLU(32, 32))
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        c1 = self.e1(x)
        c2 = self.e2(self.pool(c1))
        c3 = self.e3(self.pool(c2))
        x = self.u2(c3)
        x = self.d2(torch.cat([x, c2], dim=1))
        x = self.u1(x)
        x = self.d1(torch.cat([x, c1], dim=1))
        return self.head(x), None


class PatchEmbed(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.proj(x))


class TinyBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        y = self.norm1(t)
        t = t + self.attn(y, y, y, need_weights=False)[0]
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


class TinySegFormer(nn.Module):
    def __init__(self, num_classes, boundary=False):
        super().__init__()
        dims = [32, 64, 128]
        self.p1 = PatchEmbed(3, dims[0], 2)
        self.b1 = TinyBlock(dims[0], 2)
        self.p2 = PatchEmbed(dims[0], dims[1], 2)
        self.b2 = TinyBlock(dims[1], 4)
        self.p3 = PatchEmbed(dims[1], dims[2], 2)
        self.b3 = TinyBlock(dims[2], 4)
        self.proj1 = nn.Conv2d(dims[0], 64, 1)
        self.proj2 = nn.Conv2d(dims[1], 64, 1)
        self.proj3 = nn.Conv2d(dims[2], 64, 1)
        self.fuse = nn.Sequential(ConvBNReLU(64 * 3, 128, k=1, p=0), ConvBNReLU(128, 128))
        self.head = nn.Conv2d(128, num_classes, 1)
        self.boundary = boundary
        self.edge_head = nn.Sequential(ConvBNReLU(128, 64), nn.Conv2d(64, 1, 1)) if boundary else None

    def forward(self, x):
        h, w = x.shape[-2:]
        c1 = self.b1(self.p1(x))
        c2 = self.b2(self.p2(c1))
        c3 = self.b3(self.p3(c2))
        target = c1.shape[-2:]
        f1 = self.proj1(c1)
        f2 = F.interpolate(self.proj2(c2), size=target, mode="bilinear", align_corners=False)
        f3 = F.interpolate(self.proj3(c3), size=target, mode="bilinear", align_corners=False)
        f = self.fuse(torch.cat([f1, f2, f3], dim=1))
        logits = F.interpolate(self.head(f), size=(h, w), mode="bilinear", align_corners=False)
        edge = None
        if self.edge_head is not None:
            edge = F.interpolate(self.edge_head(f), size=(h, w), mode="bilinear", align_corners=False)
        return logits, edge


def labels_to_boundary(labels, ignore_index=VOID_CLASS):
    valid = labels != ignore_index
    b = torch.zeros_like(labels, dtype=torch.bool)
    b[:, 1:, :] |= (labels[:, 1:, :] != labels[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    b[:, :-1, :] |= (labels[:, 1:, :] != labels[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    b[:, :, 1:] |= (labels[:, :, 1:] != labels[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    b[:, :, :-1] |= (labels[:, :, 1:] != labels[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    return b.float().unsqueeze(1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, save_dir=None, codes=None):
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    edge_tp = edge_fp = edge_fn = 0
    total_correct = total_valid = 0
    times = []
    saved = 0
    for images, labels, names in loader:
        images = images.to(device)
        labels = labels.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits, edge_logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / images.size(0))
        pred = logits.argmax(1)
        valid = labels != VOID_CLASS
        total_correct += ((pred == labels) & valid).sum().item()
        total_valid += valid.sum().item()
        inds = labels[valid] * num_classes + pred[valid]
        conf += torch.bincount(inds, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

        true_b = labels_to_boundary(labels).bool()
        pred_b = labels_to_boundary(pred).bool()
        edge_tp += (pred_b & true_b).sum().item()
        edge_fp += (pred_b & ~true_b).sum().item()
        edge_fn += (~pred_b & true_b).sum().item()

        if save_dir is not None and saved < 4:
            save_dir.mkdir(parents=True, exist_ok=True)
            for i in range(min(images.size(0), 4 - saved)):
                color = colorize(pred[i].detach().cpu().numpy(), num_classes)
                Image.fromarray(color).save(save_dir / f"{Path(names[i]).stem}_pred.png")
                saved += 1

    diag = conf.diag().float()
    denom = conf.sum(1).float() + conf.sum(0).float() - diag
    iou = diag / torch.clamp(denom, min=1)
    valid_classes = torch.arange(num_classes, device=device) != VOID_CLASS
    miou = iou[valid_classes & (denom > 0)].mean().item()
    pa = total_correct / max(total_valid, 1)
    bf1 = 2 * edge_tp / max(2 * edge_tp + edge_fp + edge_fn, 1)
    fps = 1.0 / max(sum(times) / len(times), 1e-9)
    return {"mIoU": miou, "PA": pa, "BoundaryF1": bf1, "FPS": fps}


def colorize(mask, num_classes):
    rng = np.random.default_rng(123)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[VOID_CLASS] = np.array([0, 0, 0], dtype=np.uint8)
    return palette[mask.clip(0, num_classes - 1)]


def train_one(model, train_loader, test_loader, device, num_classes, epochs, lr, boundary_weight, out_dir):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(ignore_index=VOID_CLASS)
    history = []
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, edge = model(images)
            loss = ce(logits, labels)
            if edge is not None and boundary_weight > 0:
                target_edge = labels_to_boundary(labels)
                loss = loss + boundary_weight * F.binary_cross_entropy_with_logits(edge, target_edge)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        metrics = evaluate(model, test_loader, device, num_classes)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history.append(row)
        if best is None or row["mIoU"] > best["mIoU"]:
            best = row.copy()
            torch.save(model.state_dict(), out_dir / "best.pt")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    final = evaluate(model, test_loader, device, num_classes, save_dir=out_dir / "visuals")
    return best, final, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/camvid_tiny")
    parser.add_argument("--out", default="experiments/camvid_real/outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    seed_everything(42)
    root = Path(args.data)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    codes = [x.strip() for x in (root / "codes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    num_classes = len(codes)
    train_ds = CamVidTiny(root, "train")
    test_ds = CamVidTiny(root, "test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, train={len(train_ds)}, test={len(test_ds)}, classes={num_classes}")

    configs = [
        ("UNetSmall", UNetSmall(num_classes), 0.0),
        ("TinySegFormer", TinySegFormer(num_classes, boundary=False), 0.0),
        ("TinySegFormer_Boundary", TinySegFormer(num_classes, boundary=True), 0.4),
    ]
    rows = []
    for name, model, bw in configs:
        print(f"\n==> Training {name}")
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        best, final, history = train_one(model, train_loader, test_loader, device, num_classes, args.epochs, args.lr, bw, out_dir)
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        rows.append({"method": name, "best_epoch": best["epoch"], **final})

    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "best_epoch", "mIoU", "PA", "BoundaryF1", "FPS"])
        writer.writeheader()
        writer.writerows(rows)
    print("\nSummary")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
