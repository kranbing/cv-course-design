"""Training loop for semantic segmentation models."""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from experiments.config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, OUTPUT_DIR, IMAGE_SIZE, IGNORE_INDEX,
)
from experiments.utils import generate_boundary_labels, upsample_logits


def _extract_logits(output):
    """Extract logits tensor from model output (handles HF SemanticSegmenterOutput)."""
    if hasattr(output, "logits"):
        return output.logits
    return output
from experiments.metrics import compute_iou, compute_pixel_accuracy, compute_boundary_f1


def train_one_epoch(model, loader, optimizer, scaler, criterion_ce, criterion_bce,
                    boundary_lambda, epoch, writer, use_boundary=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_sem_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        with autocast(device_type=DEVICE.type):
            if use_boundary:
                sem_logits, boundary_logits = model(images)
            else:
                sem_logits = _extract_logits(model(images))
                boundary_logits = None

            # Upsample semantic logits
            sem_logits_up = upsample_logits(sem_logits, masks.shape[-2:])
            loss_ce = criterion_ce(sem_logits_up, masks)

            loss = loss_ce

            if use_boundary and boundary_logits is not None:
                # Generate boundary ground truth
                boundary_gt = generate_boundary_labels(masks).to(DEVICE)

                # Upsample boundary logits
                boundary_logits_up = upsample_logits(boundary_logits, masks.shape[-2:])

                # BCE loss for boundary
                loss_bce = criterion_bce(boundary_logits_up.squeeze(1), boundary_gt)
                loss = loss_ce + boundary_lambda * loss_bce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Track predictions for epoch mIoU
        with torch.no_grad():
            sem_pred = sem_logits_up.argmax(dim=1)
            all_sem_preds.append(sem_pred.cpu())
            all_targets.append(masks.cpu())

        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    # Compute epoch metrics
    all_preds = torch.cat(all_sem_preds)
    all_targets = torch.cat(all_targets)
    miou, _ = compute_iou(all_preds, all_targets)
    pa = compute_pixel_accuracy(all_preds, all_targets)

    avg_loss = total_loss / len(loader)

    if writer:
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Train/mIoU", miou, epoch)
        writer.add_scalar("Train/PA", pa, epoch)

    return avg_loss, miou, pa


def validate(model, loader, criterion_ce, criterion_bce, boundary_lambda,
             epoch, writer, use_boundary=True):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    all_sem_preds = []
    all_targets = []
    all_boundary_preds = []
    all_boundary_gts = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"Val Epoch {epoch}"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            if use_boundary:
                sem_logits, boundary_logits = model(images)
            else:
                sem_logits = _extract_logits(model(images))
                boundary_logits = None

            sem_logits_up = upsample_logits(sem_logits, masks.shape[-2:])
            loss_ce = criterion_ce(sem_logits_up, masks)
            loss = loss_ce

            if use_boundary and boundary_logits is not None:
                boundary_gt = generate_boundary_labels(masks).to(DEVICE)
                boundary_logits_up = upsample_logits(boundary_logits, masks.shape[-2:])
                loss_bce = criterion_bce(boundary_logits_up.squeeze(1), boundary_gt)
                loss = loss_ce + boundary_lambda * loss_bce

                # Collect boundary predictions
                boundary_pred = (torch.sigmoid(boundary_logits_up) > 0.5).float()
                all_boundary_preds.append(boundary_pred.squeeze(1).cpu())
                all_boundary_gts.append(boundary_gt.cpu())

            total_loss += loss.item()

            sem_pred = sem_logits_up.argmax(dim=1)
            all_sem_preds.append(sem_pred.cpu())
            all_targets.append(masks.cpu())

    all_preds = torch.cat(all_sem_preds)
    all_targets = torch.cat(all_targets)

    miou, ious = compute_iou(all_preds, all_targets)
    pa = compute_pixel_accuracy(all_preds, all_targets)
    avg_loss = total_loss / len(loader)

    bf1 = 0.0
    if use_boundary and all_boundary_preds:
        all_bp = torch.cat(all_boundary_preds)
        all_bgt = torch.cat(all_boundary_gts)
        bf1 = compute_boundary_f1(all_bp, all_bgt)

    if writer:
        writer.add_scalar("Val/Loss", avg_loss, epoch)
        writer.add_scalar("Val/mIoU", miou, epoch)
        writer.add_scalar("Val/PA", pa, epoch)
        if bf1 > 0:
            writer.add_scalar("Val/BoundaryF1", bf1, epoch)

    return avg_loss, miou, pa, bf1


def train_model(model, train_loader, val_loader, save_name, use_boundary=True,
                boundary_lambda=0.4, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """Full training pipeline. Returns best metrics."""
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device=DEVICE.type)

    criterion_ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_bce = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    log_dir = os.path.join(OUTPUT_DIR, "logs", save_name)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints", save_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_miou = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_pa": [], "val_bf1": []}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_miou, train_pa = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion_ce, criterion_bce,
            boundary_lambda, epoch, writer, use_boundary,
        )
        val_loss, val_miou, val_pa, val_bf1 = validate(
            model, val_loader, criterion_ce, criterion_bce,
            boundary_lambda, epoch, writer, use_boundary,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        history["val_pa"].append(val_pa)
        history["val_bf1"].append(val_bf1)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | "
              f"Val PA: {val_pa:.4f}" + (f" | Val BF1: {val_bf1:.4f}" if val_bf1 > 0 else ""))

        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "miou": val_miou,
            }, os.path.join(ckpt_dir, "best_model.pth"))

    # Save history
    with open(os.path.join(log_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    writer.close()
    print(f"\nBest mIoU: {best_miou:.4f} at epoch {best_epoch}")
    return best_miou, history
