from train_camvid_tiny import (
    CamVidTiny,
    TinySegFormer,
    UNetSmall,
    VOID_CLASS,
    evaluate,
    seed_everything,
    train_one,
)

import argparse
import csv
import json
import random
from pathlib import Path

from torch.utils.data import DataLoader
import torch


class CamVidBenchmark(CamVidTiny):
    def __init__(self, root: Path, split: str, image_size=(96, 128)):
        self.root = Path(root)
        self.image_size = image_size
        self.split = split
        valid_names = set((self.root / "valid.txt").read_text(encoding="utf-8").splitlines())
        images = sorted((self.root / "images").glob("*.png"))
        pairs = []
        for img in images:
            lab = self.root / "labels" / f"{img.stem}_P.png"
            if not lab.exists():
                continue
            is_valid = img.name in valid_names
            if (split == "train" and not is_valid) or (split in {"valid", "test"} and is_valid):
                pairs.append((img, lab))
        self.pairs = pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/camvid")
    parser.add_argument("--out", default="experiments/camvid_benchmark/outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()

    seed_everything(42)
    root = Path(args.data)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    codes = [x.strip() for x in (root / "codes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    num_classes = len(codes)
    image_size = (args.height, args.width)
    train_ds = CamVidBenchmark(root, "train", image_size=image_size)
    test_ds = CamVidBenchmark(root, "test", image_size=image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device}, train={len(train_ds)}, test={len(test_ds)}, "
        f"classes={num_classes}, image_size={image_size}"
    )

    configs = [
        ("UNetSmall", UNetSmall(num_classes), 0.0),
        ("TinySegFormer", TinySegFormer(num_classes, boundary=False), 0.0),
        ("TinySegFormer_Boundary", TinySegFormer(num_classes, boundary=True), 0.4),
    ]
    rows = []
    for name, model, boundary_weight in configs:
        print(f"\n==> Training {name}")
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        best, final, history = train_one(
            model,
            train_loader,
            test_loader,
            device,
            num_classes,
            args.epochs,
            args.lr,
            boundary_weight,
            out_dir,
        )
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        rows.append({"method": name, "best_epoch": best["epoch"], **final})

    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "best_epoch", "mIoU", "PA", "BoundaryF1", "FPS"])
        writer.writeheader()
        writer.writerows(rows)
    with (out_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(root),
                "train_count": len(train_ds),
                "test_count": len(test_ds),
                "classes": num_classes,
                "ignore_index": VOID_CLASS,
                "image_size": [args.height, args.width],
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "split": "fastai CamVid valid.txt",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\nSummary")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
