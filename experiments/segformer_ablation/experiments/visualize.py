"""Visualization: training curves, comparison charts, qualitative results."""
import os
import json
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from experiments.config import OUTPUT_DIR, DATA_DIR, DEVICE, NUM_CLASSES, IMAGE_SIZE, CAMVID_CLASSES
from experiments.dataset import get_dataloaders, _imread
from experiments.utils import generate_boundary_labels, upsample_logits

# Use a clean style
rcParams["font.family"] = "DejaVu Sans"
rcParams["font.size"] = 11

CLS_NAMES = CAMVID_CLASSES + ["void"]
COLORS = plt.cm.tab20(np.linspace(0, 1, len(CLS_NAMES)))


def plot_training_curves(history_path, save_path=None):
    """Plot training and validation loss, mIoU curves."""
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Loss
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train Loss", linewidth=1.2)
    ax.plot(history["val_loss"], label="Val Loss", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # mIoU
    ax = axes[1]
    ax.plot(history["val_miou"], label="Val mIoU", color="green", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.set_title("Validation mIoU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PA
    ax = axes[2]
    ax.plot(history["val_pa"], label="Val PA", color="orange", linewidth=1.5)
    if any(v > 0 for v in history.get("val_bf1", [0])):
        ax.plot(history["val_bf1"], label="Val Boundary F1", color="red", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Pixel Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_comparison_bar(results_dict, save_path=None):
    """Plot mIoU and Boundary F1 comparison bar chart."""
    models = list(results_dict.keys())
    mious = [results_dict[m].get("mIoU", 0) for m in models]
    bfs = [results_dict[m].get("Boundary_F1", 0) for m in models]
    fps = [results_dict[m].get("FPS", 0) for m in models]

    # Short display names
    display_names = {
        "unet": "UNet", "deeplabv3": "DeepLabV3+",
        "segformer": "SegFormer-B0",
        "segformer_boundary_lambda0.0": "+Boundary λ=0",
        "segformer_boundary_lambda0.2": "+Boundary λ=0.2",
        "segformer_boundary_lambda0.4": "+Boundary λ=0.4",
        "segformer_boundary_lambda0.6": "+Boundary λ=0.6",
    }
    names = [display_names.get(m, m) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    # mIoU
    ax = axes[0]
    bars = ax.bar(names, mious, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Semantic Segmentation mIoU Comparison")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    for bar, v in zip(bars, mious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # Boundary F1
    ax = axes[1]
    bars = ax.bar(names, bfs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Boundary F1 (%)")
    ax.set_title("Boundary Prediction F1 Score")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    for bar, v in zip(bars, bfs):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_segmentation(model, dataloader, save_dir, use_boundary=True, num_samples=4):
    """Generate qualitative segmentation comparison images."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            if idx >= num_samples:
                break

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            if use_boundary:
                sem_logits, boundary_logits = model(images)
            else:
                raw = model(images)
                sem_logits = raw.logits if hasattr(raw, "logits") else raw
                boundary_logits = None

            sem_up = upsample_logits(sem_logits, masks.shape[-2:])
            pred = sem_up.argmax(dim=1)

            for b in range(images.shape[0]):
                img = images[b].cpu().permute(1, 2, 0).numpy()
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img, 0, 1)

                gt = masks[b].cpu().numpy()
                pred_mask = pred[b].cpu().numpy()

                fig, axes = plt.subplots(1, 4 if boundary_logits is not None else 3,
                                         figsize=(16, 4))
                axes[0].imshow(img)
                axes[0].set_title("Input Image")
                axes[0].axis("off")

                axes[1].imshow(gt, vmin=0, vmax=NUM_CLASSES - 1, cmap="tab20")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(pred_mask, vmin=0, vmax=NUM_CLASSES - 1, cmap="tab20")
                axes[2].set_title("Prediction")
                axes[2].axis("off")

                if boundary_logits is not None:
                    bd = torch.sigmoid(upsample_logits(boundary_logits, masks.shape[-2:]))
                    axes[3].imshow(bd[b, 0].cpu().numpy(), cmap="hot")
                    axes[3].set_title("Boundary Prediction")
                    axes[3].axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"sample_{idx}_{b}.png"),
                            dpi=150, bbox_inches="tight")
                plt.close()


def generate_all_visualizations():
    """Generate all figures for the report."""
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Training curves
    log_dir = os.path.join(OUTPUT_DIR, "logs")
    for model_name in os.listdir(log_dir):
        hist_path = os.path.join(log_dir, model_name, "history.json")
        if os.path.exists(hist_path):
            plot_training_curves(hist_path, os.path.join(fig_dir, f"curves_{model_name}.png"))
            print(f"  Training curves: {model_name}")

    # 2. Results comparison
    results_dir = os.path.join(OUTPUT_DIR, "results")
    results = {}
    for f in os.listdir(results_dir):
        if f.endswith("_test.pt"):
            name = f.replace("_test.pt", "")
            results[name] = torch.load(os.path.join(results_dir, f), map_location="cpu", weights_only=False)
    if results:
        plot_comparison_bar(results, os.path.join(fig_dir, "comparison_bar.png"))
        print("  Comparison bar chart")

    # 3. Qualitative segmentation examples
    _, _, test_loader = get_dataloaders(batch_size=1, num_workers=0)

    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    for model_name in ["segformer", "segformer_boundary_lambda0.4"]:
        ckpt_path = os.path.join(ckpt_dir, model_name, "best_model.pth")
        if not os.path.exists(ckpt_path):
            continue
        print(f"  Qualitative results: {model_name}")

        if "boundary" in model_name:
            from experiments.models.segformer_boundary import SegFormerBoundary
            model = SegFormerBoundary(num_classes=NUM_CLASSES)
            ub = True
        else:
            from transformers import SegformerForSemanticSegmentation
            model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
            )
            ub = False

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(DEVICE)

        visualize_segmentation(model, test_loader,
                               os.path.join(fig_dir, f"qualitative_{model_name}"),
                               use_boundary=ub, num_samples=4)

    print("Visualizations generated in:", fig_dir)


if __name__ == "__main__":
    generate_all_visualizations()
