"""Evaluation on test set: mIoU, PA, Boundary F1, FPS, Params."""
import os
import torch
from tqdm import tqdm

from experiments.config import DEVICE, NUM_CLASSES, IMAGE_SIZE, OUTPUT_DIR
from experiments.utils import generate_boundary_labels, upsample_logits, count_parameters
from experiments.metrics import compute_iou, compute_pixel_accuracy, compute_boundary_f1, measure_fps


def evaluate_test(model, test_loader, use_boundary=True, boundary_threshold=0.5,
                  save_visualizations=False, save_name="model"):
    """Full evaluation on test set."""
    model = model.to(DEVICE)
    model.eval()

    all_sem_preds = []
    all_targets = []
    all_boundary_preds = []
    all_boundary_gts = []
    vis_images = []
    vis_masks = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            if use_boundary:
                sem_logits, boundary_logits = model(images)
            else:
                raw = model(images)
                sem_logits = raw.logits if hasattr(raw, "logits") else raw
                boundary_logits = None

            sem_logits_up = upsample_logits(sem_logits, masks.shape[-2:])
            sem_pred = sem_logits_up.argmax(dim=1)

            all_sem_preds.append(sem_pred.cpu())
            all_targets.append(masks.cpu())

            # Save for visualization
            if save_visualizations and len(vis_images) < 5:
                vis_images.append(images[0:1].cpu())
                vis_masks.append({
                    "gt": masks[0].cpu(),
                    "pred": sem_pred[0].cpu(),
                })

            if use_boundary and boundary_logits is not None:
                boundary_gt = generate_boundary_labels(masks).to(DEVICE)
                boundary_logits_up = upsample_logits(boundary_logits, masks.shape[-2:])
                boundary_pred = (torch.sigmoid(boundary_logits_up) > boundary_threshold).float()
                all_boundary_preds.append(boundary_pred.squeeze(1).cpu())
                all_boundary_gts.append(boundary_gt.cpu())

    all_preds = torch.cat(all_sem_preds)
    all_targets = torch.cat(all_targets)

    miou, ious = compute_iou(all_preds, all_targets)
    pa = compute_pixel_accuracy(all_preds, all_targets)

    result = {
        "mIoU": miou * 100,
        "PA": pa * 100,
        "per_class_iou": {i: ious[i] * 100 for i in range(len(ious))},
        "Params_M": count_parameters(model),
    }

    if all_boundary_preds:
        all_bp = torch.cat(all_boundary_preds)
        all_bgt = torch.cat(all_boundary_gts)
        bf1 = compute_boundary_f1(all_bp, all_bgt)
        result["Boundary_F1"] = bf1 * 100
    else:
        result["Boundary_F1"] = 0.0

    # Measure FPS
    fps = measure_fps(model, input_size=IMAGE_SIZE, device=DEVICE)
    result["FPS"] = fps

    print(f"\n{'='*50}")
    print(f"Test Results for {save_name}:")
    print(f"  mIoU: {result['mIoU']:.2f}%")
    print(f"  PA:   {result['PA']:.2f}%")
    if result["Boundary_F1"] > 0:
        print(f"  BF1:  {result['Boundary_F1']:.2f}%")
    print(f"  Params: {result['Params_M']:.1f}M")
    print(f"  FPS: {result['FPS']:.1f}")
    print(f"{'='*50}\n")

    # Save results
    os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)
    torch.save(result, os.path.join(OUTPUT_DIR, "results", f"{save_name}_test.pt"))

    return result
