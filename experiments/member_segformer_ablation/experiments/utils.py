"""Utility functions: boundary label generation, seed setting, etc."""
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def generate_boundary_labels(semantic_masks, kernel_size=3, num_classes=11):
    """Generate binary boundary labels from semantic segmentation masks.

    Uses morphological dilation - erosion to extract class boundaries.
    Works with both numpy arrays and torch tensors.

    Args:
        semantic_masks: (B, H, W) tensor of class indices or (H, W) numpy array
        kernel_size: Morphological kernel size for boundary extraction
        num_classes: Number of semantic classes

    Returns:
        binary_boundary: (B, H, W) tensor of 0/1, 1 = boundary pixel
    """
    if isinstance(semantic_masks, torch.Tensor):
        semantic_masks = semantic_masks.cpu().numpy()

    single_input = semantic_masks.ndim == 2
    if single_input:
        semantic_masks = semantic_masks[np.newaxis, ...]

    B, H, W = semantic_masks.shape
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    boundaries = np.zeros((B, H, W), dtype=np.float32)

    for b in range(B):
        mask = semantic_masks[b].astype(np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundaries[b] = (dilated != eroded).astype(np.float32)

    if single_input:
        boundaries = boundaries[0]

    return torch.from_numpy(boundaries)


def upsample_logits(logits, target_size):
    """Upsample logits to target spatial size."""
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def count_parameters(model):
    """Return model parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6
