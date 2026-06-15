"""Main entry point: train and evaluate one model configuration.

Usage:
    python main.py --model unet
    python main.py --model deeplabv3
    python main.py --model segformer
    python main.py --model segformer_boundary --lambda 0.4
    python main.py --model segformer_boundary --lambda 0.0   # ablation: no boundary loss
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))

from experiments.config import DEVICE, BATCH_SIZE, NUM_CLASSES, OUTPUT_DIR
from experiments.dataset import download_camvid, get_dataloaders
from experiments.models.unet import UNet
from experiments.models.deeplabv3 import create_deeplabv3
from experiments.models.segformer_boundary import SegFormerBoundary
from experiments.train import train_model
from experiments.eval import evaluate_test
from experiments.utils import set_seed


def get_model(model_name):
    """Create model based on name."""
    if model_name == "unet":
        model = UNet(num_classes=NUM_CLASSES)
        use_boundary = False
    elif model_name == "deeplabv3":
        model = create_deeplabv3(num_classes=NUM_CLASSES)
        use_boundary = False
    elif model_name == "segformer":
        from transformers import SegformerForSemanticSegmentation
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
        )
        use_boundary = False
    elif model_name == "segformer_boundary":
        model = SegFormerBoundary(num_classes=NUM_CLASSES)
        use_boundary = True
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model, use_boundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "deeplabv3", "segformer", "segformer_boundary"])
    parser.add_argument("--lambda", type=float, default=0.4, dest="boundary_lambda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Model: {args.model}, Lambda: {args.boundary_lambda}, Classes: {NUM_CLASSES}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    model, use_boundary = get_model(args.model)

    if args.model == "segformer_boundary":
        save_name = f"segformer_boundary_lambda{args.boundary_lambda}"
    else:
        save_name = args.model

    print(f"\nTraining {save_name}...")
    best_miou, history = train_model(
        model, train_loader, val_loader, save_name=save_name,
        use_boundary=use_boundary, boundary_lambda=args.boundary_lambda,
        num_epochs=args.epochs, lr=args.lr,
    )

    # Load best checkpoint and evaluate
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", save_name, "best_model.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    results = evaluate_test(model, test_loader, use_boundary=use_boundary, save_name=save_name)
    return results


if __name__ == "__main__":
    main()
