import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Reuse your existing architecture + metrics
from model import UNet, DiceLoss, compute_miou


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism helps reproducibility, but can slow things down.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_int_list(s: str) -> List[int]:
    # Accept "1,2,3" or "1 2 3"
    s = s.replace(",", " ")
    return [int(x) for x in s.split() if x.strip()]


class SegmentationDataset(Dataset):
    """
    Generic segmentation dataset:
    - images: img_dir/*.jpg (or chosen extension)
    - masks:  mask_dir/<same_stem>*.png (or chosen extension)
    - mask pixels are mapped to class indices via `binary_foreground_values` or `class_pixel_values`.
    """

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        img_ext: str,
        mask_ext: str,
        transform=None,
        num_classes: int = 2,
        binary_foreground_values: Optional[List[int]] = None,
        class_pixel_values: Optional[List[int]] = None,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.num_classes = num_classes

        self.binary_foreground_values = binary_foreground_values or []
        self.class_pixel_values = class_pixel_values or []

        img_glob = os.path.join(img_dir, f"*.{img_ext}")
        self.img_paths = sorted([p for p in glob_glob(img_glob)])
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found with pattern: {img_glob}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_mask(self, mask_path: str) -> torch.Tensor:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found/readable: {mask_path}")

        # Map mask pixel values to class indices.
        if self.num_classes == 2:
            out = np.zeros_like(mask, dtype=np.uint8)
            for v in self.binary_foreground_values:
                out[mask == v] = 1
            return torch.from_numpy(out).long()

        # Multiclass
        if len(self.class_pixel_values) != self.num_classes:
            raise ValueError(
                f"Expected `class_pixel_values` length {self.num_classes}, got {len(self.class_pixel_values)}"
            )
        out = np.zeros_like(mask, dtype=np.uint8)
        for class_idx, pixel_value in enumerate(self.class_pixel_values):
            out[mask == pixel_value] = class_idx
        return torch.from_numpy(out).long()

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{stem}.{self.mask_ext}")

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if img is None:
            raise FileNotFoundError(f"Image not found/readable: {img_path}")

        mask = self._load_mask(mask_path)

        if self.transform is not None:
            # albumentations expects numpy mask
            aug = self.transform(image=img, mask=mask.numpy())
            img = aug["image"]
            mask = aug["mask"]

            # ToTensorV2 makes mask a torch tensor already.
            if isinstance(mask, torch.Tensor):
                return img, mask.long()
            # Some albumentations versions return mask as numpy.
            return img, torch.from_numpy(mask).long()

        return img, mask.long()


def glob_glob(pattern: str):
    # Simple wrapper to avoid importing glob in multiple places.
    import glob

    return glob.glob(pattern)


def make_transforms(img_h: int, img_w: int):
    train_transform = A.Compose(
        [
            A.Resize(img_h, img_w),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=10, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(img_h, img_w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return train_transform, val_transform


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes)

    def forward(self, preds, targets):
        # preds: [B, C, H, W], targets: [B, H, W] in {0..C-1}
        return self.ce(preds, targets) + self.dice(preds, targets)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    miou_sum = 0.0
    count = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        miou_sum += compute_miou(logits, masks, num_classes=num_classes)
        count += 1
    return miou_sum / max(count, 1)


def benchmark_fps(model, device, img_h: int, img_w: int, warmup: int = 20, iters: int = 100):
    model.eval()
    x = torch.randn(1, 3, img_h, img_w, device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    if device.type == "cuda":
        start.record()

    for _ in range(iters):
        with torch.no_grad():
            _ = model(x)
    if device.type == "cuda":
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        fps = 1000.0 / ms
        return fps, ms

    # CPU timing fallback
    import time

    t0 = time.time()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(x)
    t1 = time.time()
    total = t1 - t0
    ms = (total * 1000.0) / iters
    fps = 1000.0 / ms
    return fps, ms


def main():
    parser = argparse.ArgumentParser(description="Train drivable space segmentation from scratch.")
    parser.add_argument("--train-img-dir", required=True)
    parser.add_argument("--train-mask-dir", required=True)
    parser.add_argument("--val-img-dir", required=True)
    parser.add_argument("--val-mask-dir", required=True)
    parser.add_argument("--img-ext", default="jpg")
    parser.add_argument("--mask-ext", default="png")

    parser.add_argument("--num-classes", type=int, default=2, choices=[2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--img-height", type=int, default=256)
    parser.add_argument("--img-width", type=int, default=512)

    # Binary mapping (recommended for hackathon)
    parser.add_argument(
        "--binary-foreground-values",
        type=str,
        default="1",
        help="Comma/space-separated mask pixel values that should map to class 1 (drivable). Everything else becomes class 0.",
    )

    # Multi-class mapping
    parser.add_argument(
        "--class-pixel-values",
        type=str,
        default="",
        help="If num_classes > 2, give comma/space-separated mask pixel values for classes 0..C-1.",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save-dir", default="outputs")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use full train set.")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    binary_foreground_values = parse_int_list(args.binary_foreground_values)
    class_pixel_values = parse_int_list(args.class_pixel_values) if args.class_pixel_values else []

    train_transform, val_transform = make_transforms(args.img_height, args.img_width)

    train_ds = SegmentationDataset(
        img_dir=args.train_img_dir,
        mask_dir=args.train_mask_dir,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        transform=train_transform,
        num_classes=args.num_classes,
        binary_foreground_values=binary_foreground_values,
        class_pixel_values=class_pixel_values,
    )
    if args.max_train_samples and args.max_train_samples > 0:
        # Quick debug mode for the deadline.
        train_ds.img_paths = train_ds.img_paths[: args.max_train_samples]

    val_ds = SegmentationDataset(
        img_dir=args.val_img_dir,
        mask_dir=args.val_mask_dir,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        transform=val_transform,
        num_classes=args.num_classes,
        binary_foreground_values=binary_foreground_values,
        class_pixel_values=class_pixel_values,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet(num_classes=args.num_classes).to(device)
    criterion = CombinedLoss(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_miou = -1.0
    best_path = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, (images, masks) in enumerate(pbar, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if args.log_every and step % args.log_every == 0:
                pbar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)

        val_miou = evaluate(model, val_loader, device, num_classes=args.num_classes)
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f} val_mIoU={val_miou:.4f}")

        # Save best checkpoint for submission/repro.
        if val_miou > best_miou:
            best_miou = val_miou
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": args.num_classes,
                    "img_height": args.img_height,
                    "img_width": args.img_width,
                    "binary_foreground_values": binary_foreground_values,
                    "class_pixel_values": class_pixel_values,
                },
                best_path,
            )

    fps, ms = benchmark_fps(model, device, args.img_height, args.img_width)
    metrics = {
        "best_val_mIoU": best_miou,
        "fps_benchmark": fps,
        "ms_per_frame": ms,
        "device": str(device),
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }

    with open(os.path.join(args.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

