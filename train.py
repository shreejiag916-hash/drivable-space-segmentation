"""
================================================
 Training Script — Real-time Drivable Space Segmentation
 Model   : Custom U-Net (trained from scratch)
 Dataset : BDD100K (External)
 MAHE Mobility Hackathon 2026 — Track 2
================================================

Usage:
    python train.py --data_root /path/to/bdd100k --epochs 25 --batch_size 8
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from dataset import get_dataloaders
from model import UNet

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Train] Using device: {device}")


# ── Loss Functions ────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        loss = 0
        for c in range(self.num_classes):
            inter = (preds[:, c] * (targets == c).float()).sum()
            union = preds[:, c].sum() + (targets == c).float().sum()
            loss += 1 - (2 * inter + self.smooth) / (union + self.smooth)
        return loss / self.num_classes


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=3)

    def forward(self, preds, targets):
        return self.ce(preds, targets) + self.dice(preds, targets)


# ── Metric ────────────────────────────────────────────────────────────────────
def compute_miou(preds, targets, num_classes=3):
    preds = torch.argmax(preds, dim=1)
    iou_list = []
    for c in range(num_classes):
        inter = ((preds == c) & (targets == c)).sum().float()
        union = ((preds == c) | (targets == c)).sum().float()
        if union > 0:
            iou_list.append((inter / union).item())
    return sum(iou_list) / len(iou_list) if iou_list else 0.0


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, total_miou = 0.0, 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast():
            preds = model(imgs)
            loss  = criterion(preds, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_miou += compute_miou(preds.detach(), masks)
    n = len(loader)
    return total_loss / n, total_miou / n


# ── Validate ──────────────────────────────────────────────────────────────────
def validate(model, loader, criterion):
    model.eval()
    total_loss, total_miou = 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            total_loss += criterion(preds, masks).item()
            total_miou += compute_miou(preds, masks)
    n = len(loader)
    return total_loss / n, total_miou / n


# ── Plot curves ───────────────────────────────────────────────────────────────
def save_plots(train_ious, val_ious, train_losses, val_losses, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_ious) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_ious,   label="Train mIoU")
    axes[0].plot(epochs, val_ious,     label="Val mIoU")
    axes[0].set_title("mIoU per Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("mIoU")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, train_losses, label="Train Loss")
    axes[1].plot(epochs, val_losses,   label="Val Loss")
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[Train] Curves saved → {out_dir}/training_curves.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    train_loader, val_loader = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )

    model     = UNet(num_classes=3).to(device)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()

    best_miou = 0.0
    train_ious, val_ious     = [], []
    train_losses, val_losses = [], []

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_miou = train_epoch(model, train_loader, optimizer, criterion, scaler)
        vl_loss, vl_miou = validate(model, val_loader, criterion)
        scheduler.step()

        train_ious.append(tr_miou);   val_ious.append(vl_miou)
        train_losses.append(tr_loss); val_losses.append(vl_loss)

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"Train Loss: {tr_loss:.4f}  mIoU: {tr_miou:.4f} | "
            f"Val Loss: {vl_loss:.4f}  mIoU: {vl_miou:.4f} | "
            f"Time: {time.time()-t0:.1f}s"
        )

        if vl_miou > best_miou:
            best_miou = vl_miou
            ckpt = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  ✔ New best mIoU: {best_miou:.4f} — saved to {ckpt}")

    save_plots(train_ious, val_ious, train_losses, val_losses, args.save_dir)
    print(f"\n[Done] Best Val mIoU: {best_miou:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for Drivable Space Segmentation")
    parser.add_argument("--data_root",   type=str, required=True, help="Path to BDD100K root")
    parser.add_argument("--epochs",      type=int, default=25)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir",    type=str, default="checkpoints")
    args = parser.parse_args()
    main(args)
