"""CamVid dataset loader with augmentations."""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import albumentations as A
import numpy as np
import urllib.request

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.config import DATA_DIR, CAMVID_MEAN, CAMVID_STD, IMAGE_SIZE

CAMVID_BASE_URL = "https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid"
CAMVID_API_URL = "https://api.github.com/repos/alexgkendall/SegNet-Tutorial/contents/CamVid"


def _imread(path):
    """Read image via binary buffer (avoids OpenCV unicode path issues on Windows)."""
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _list_github_dir(dir_path):
    """List files in a GitHub directory via the API."""
    url = f"{CAMVID_API_URL}/{dir_path}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    with urllib.request.urlopen(req) as resp:
        import json
        return json.loads(resp.read())


def _download_file(url, dest_path):
    """Download a single file."""
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"  Failed: {url} — {e}")
        return False


def download_camvid(max_workers=8):
    """Download CamVid dataset from GitHub if not present."""
    splits = {
        "train": "train", "trainannot": "trainannot",
        "val": "val", "valannot": "valannot",
        "test": "test", "testannot": "testannot",
    }

    all_done = True
    for local_name in splits.keys():
        if not os.path.exists(os.path.join(DATA_DIR, local_name)):
            all_done = False
            break
    if all_done:
        print(f"CamVid already exists at {DATA_DIR}")
        # Quick check we actually have files
        train_files = os.listdir(os.path.join(DATA_DIR, "train"))
        if len(train_files) > 0:
            print(f"  Found {len(train_files)} training images")
            return

    os.makedirs(DATA_DIR, exist_ok=True)

    for local_name, gh_name in splits.items():
        local_dir = os.path.join(DATA_DIR, local_name)
        if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
            print(f"  {local_name}/ already has files, skipping")
            continue
        os.makedirs(local_dir, exist_ok=True)

        print(f"Listing CamVid/{gh_name}/ ...")
        try:
            entries = _list_github_dir(gh_name)
        except Exception as e:
            print(f"  Failed to list {gh_name}: {e}")
            continue

        files_to_dl = [(e["name"], e["download_url"]) for e in entries
                       if e["type"] == "file" and e["name"].endswith((".png", ".jpg", ".PNG", ".JPG"))]

        if not files_to_dl:
            # Try without extension filter
            files_to_dl = [(e["name"], e["download_url"]) for e in entries if e["type"] == "file"]

        print(f"  Downloading {len(files_to_dl)} files to {local_name}/ ...")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for name, url in files_to_dl:
                dest = os.path.join(local_dir, name)
                futures[pool.submit(_download_file, url, dest)] = name

            done = 0
            for future in as_completed(futures):
                done += 1
                if done % 50 == 0:
                    print(f"    {local_name}: {done}/{len(files_to_dl)}")

        print(f"  {local_name}/ complete: {len(os.listdir(local_dir))} files")

    print("CamVid download complete.")


def load_camvid_paths(split="train"):
    """Return lists of (image_path, mask_path) for the given split."""
    img_dir = os.path.join(DATA_DIR, split)
    if split == "train":
        mask_dir = os.path.join(DATA_DIR, "trainannot")
    elif split == "val":
        mask_dir = os.path.join(DATA_DIR, "valannot")
    else:
        mask_dir = os.path.join(DATA_DIR, "testannot")

    img_paths = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".PNG", ".JPG"))
    ])
    mask_paths = sorted([
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir) if f.endswith((".png", ".jpg", ".PNG", ".JPG"))
    ])

    assert len(img_paths) == len(mask_paths), f"Mismatch: {len(img_paths)} images vs {len(mask_paths)} masks in {split}"
    return list(zip(img_paths, mask_paths))


# CamVid color map (BGR order for OpenCV — cv2 reads as BGR)
CAMVID_COLORMAP = {
    (128, 128, 128): 0,    # sky
    (128, 0, 0): 1,        # building
    (192, 192, 128): 2,    # pole
    (128, 64, 128): 3,     # road
    (0, 0, 192): 4,        # pavement
    (128, 128, 0): 5,      # tree
    (192, 128, 128): 6,    # signsymbol
    (64, 64, 128): 7,      # fence
    (64, 0, 128): 8,       # car
    (64, 64, 0): 9,        # pedestrian
    (0, 128, 192): 10,     # bicyclist
}


def encode_mask(color_mask):
    """Convert mask image to class index map.

    Handles both RGB colormap masks and grayscale class-indexed masks.
    """
    # Check if already class-indexed (all channels equal = grayscale encoding)
    if np.array_equal(color_mask[:, :, 0], color_mask[:, :, 1]) and \
       np.array_equal(color_mask[:, :, 0], color_mask[:, :, 2]):
        return color_mask[:, :, 0].astype(np.int64)

    # Fallback: RGB colormap encoding
    h, w = color_mask.shape[:2]
    label = np.full((h, w), 255, dtype=np.int64)
    for bgr, cls in CAMVID_COLORMAP.items():
        match = np.all(color_mask == np.array(bgr, dtype=np.uint8), axis=-1)
        label[match] = cls
    return label


class CamVidDataset(Dataset):
    """CamVid semantic segmentation dataset."""

    def __init__(self, split="train", augment=False):
        self.paths = load_camvid_paths(split)
        self.split = split
        self.augment = augment and split == "train"

        self.norm = transforms.Normalize(mean=CAMVID_MEAN, std=CAMVID_STD)

        if self.augment:
            self.transform = A.Compose([
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.PadIfNeeded(min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1],
                              border_mode=cv2.BORDER_REFLECT_101),
                A.RandomCrop(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path, mask_path = self.paths[idx]

        image = _imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_color = _imread(mask_path)
        if mask_color is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        if self.transform:
            transformed = self.transform(image=image, mask=mask_color)
            image = transformed["image"]
            mask_color = transformed["mask"]

        mask = encode_mask(mask_color)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = self.norm(image_tensor)
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor


def get_dataloaders(batch_size=8, num_workers=2):
    """Return train/val/test dataloaders."""
    from torch.utils.data import DataLoader

    train_ds = CamVidDataset("train", augment=True)
    val_ds = CamVidDataset("val", augment=False)
    test_ds = CamVidDataset("test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
