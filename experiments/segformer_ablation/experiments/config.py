"""Global configuration for all experiments."""
import os
import torch

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(ROOT))
DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "CamVid_member_segformer")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(ROOT, "data", "CamVid")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CamVid dataset
CAMVID_CLASSES = [
    "sky", "building", "pole", "road", "pavement",
    "tree", "signsymbol", "fence", "car", "pedestrian", "bicyclist",
]
NUM_CLASSES = 12  # 11 semantic classes + void (class 11)
IGNORE_INDEX = 11   # class 11 is void/unlabeled
CAMVID_MEAN = [0.485, 0.456, 0.406]
CAMVID_STD = [0.229, 0.224, 0.225]

# Training
BATCH_SIZE = 12  # increased from 8; 16 was too close to VRAM limit
NUM_EPOCHS = 80
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.01
IMAGE_SIZE = (352, 480)  # height divisible by 32 (UNet+SegFormer requirement)

# Boundary loss weight (default for our model)
BOUNDARY_LAMBDA = 0.4

# Random seed
SEED = 42
