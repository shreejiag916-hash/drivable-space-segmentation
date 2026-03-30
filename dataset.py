"""
================================================
 BDD100K Dataset Loader
 Real-time Drivable Space Segmentation
 MAHE Mobility Hackathon 2026 — Track 2
================================================

Expected folder structure:
    bdd100k/
    ├── images/
    │   ├── train/   *.jpg
    │   └── val/     *.jpg
    └── drivable_maps/
        ├── train/   *.png
        └── val/     *.png

Mask pixel values:
    0   → Background (non-drivable)
    127 → Drivable area (main lane)
    191 → Drivable area (alternative lane)
"""

import os
import glob
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


# ── Label remapping ──────────────────────────────────────────────────────────
def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Convert BDD100K raw mask pixel values to class indices 0/1/2."""
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask == 127] = 1   # main drivable lane
    out[mask == 191] = 2   # alternative drivable lane
    return out


# ── Augmentation pipelines ───────────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(256, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=10, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ── Dataset class ─────────────────────────────────────────────────────────────
class BDDDataset(Dataset):
    """
    PyTorch Dataset for BDD100K drivable-area segmentation.

    Args:
        img_dir  (str): Path to directory containing input images (*.jpg).
        mask_dir (str): Path to directory containing segmentation masks (*.png).
        transform: Albumentations transform pipeline.
    """

    def __init__(self, img_dir: str, mask_dir: str, transform=None):
        self.img_paths  = sorted(glob.glob(os.path.join(img_dir,  "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform  = transform

        assert len(self.img_paths) == len(self.mask_paths), (
            f"Image/mask count mismatch: {len(self.img_paths)} vs {len(self.mask_paths)}"
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img  = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = preprocess_mask(
            cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        )

        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        return img, mask.long()


# ── DataLoader factory ────────────────────────────────────────────────────────
def get_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
):
    """
    Build train and validation DataLoaders for BDD100K.

    Args:
        data_root  (str): Root path containing images/ and drivable_maps/.
        batch_size (int): Samples per batch.
        num_workers(int): Parallel data-loading workers.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    train_ds = BDDDataset(
        img_dir  = os.path.join(data_root, "images", "train"),
        mask_dir = os.path.join(data_root, "drivable_maps", "train"),
        transform = train_transform,
    )
    val_ds = BDDDataset(
        img_dir  = os.path.join(data_root, "images", "val"),
        mask_dir = os.path.join(data_root, "drivable_maps", "val"),
        transform = val_transform,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    print(f"[Dataset] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader
