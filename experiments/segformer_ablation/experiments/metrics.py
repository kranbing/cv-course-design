"""Evaluation metrics: mIoU, Pixel Accuracy, Boundary F1, FPS."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from experiments.config import NUM_CLASSES, IGNORE_INDEX, DEVICE


def compute_iou(pred, target, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    """Compute per-class and mean IoU.

    Args:
        pred: (N, H, W) tensor of predicted class indices
        target: (N, H, W) tensor of ground truth class indices
    Returns:
        miou: float, mean IoU over valid classes
        ious: list of per-class IoU values
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
        else:
            ious.append(float("nan"))
    valid = [v for v in ious if not np.isnan(v)]
    miou = np.mean(valid) if valid else 0.0
    return miou, ious


def compute_pixel_accuracy(pred, target, ignore_index=IGNORE_INDEX):
    """Compute pixel accuracy, excluding ignored pixels."""
    valid = (target != ignore_index)
    correct = (pred[valid] == target[valid]).sum().float()
    total = valid.sum().float()
    return (correct / total).item() if total > 0 else 0.0


def compute_boundary_f1(pred_boundary, gt_boundary):
    """Compute F1 score for boundary prediction.

    Args:
        pred_boundary: (N, H, W) binary predictions (0/1) or logits
        gt_boundary: (N, H, W) binary ground truth (0/1)
    """
    if pred_boundary.dtype != torch.bool:
        pred_boundary = (torch.sigmoid(pred_boundary) > 0.5).float()
    pred_flat = pred_boundary.cpu().numpy().flatten().astype(np.int32)
    gt_flat = gt_boundary.cpu().numpy().flatten().astype(np.int32)
    return f1_score(gt_flat, pred_flat, zero_division=0)


def measure_fps(model, input_size=(360, 480), device=DEVICE, num_warmup=10, num_runs=100):
    """Measure inference FPS on the given device."""
    model.eval()
    dummy = torch.randn(1, 3, *input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)

    # Timed runs
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.time()

    fps = num_runs / (t1 - t0)
    return fps
