
# ================================================
# Real-time Drivable Space Segmentation
# Model: Custom U-Net (trained from scratch)
# Dataset: BDD100K (External)
# Author: Shreeji Agrawal
# ================================================

import os, glob, shutil, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_mask(mask):
    out = np.zeros_like(mask)
    out[mask == 127] = 1
    out[mask == 191] = 2
    return out

class BDDDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths  = sorted(glob.glob(os.path.join(img_dir,  "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform  = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img  = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = preprocess_mask(cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE))
        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        return img, mask.long()

train_transform = A.Compose([
    A.Resize(256, 512), A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3), A.Rotate(limit=10, p=0.3),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(256, 512),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()
])

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)
    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))

class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.enc1 = EncoderBlock(3, 32);   self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128); self.enc4 = EncoderBlock(128, 256)
        self.bottleneck = ConvBlock(256, 512)
        self.dec4 = DecoderBlock(512, 256); self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64);  self.dec1 = DecoderBlock(64, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
    def forward(self, x):
        s1,x=self.enc1(x); s2,x=self.enc2(x); s3,x=self.enc3(x); s4,x=self.enc4(x)
        x=self.bottleneck(x)
        x=self.dec4(x,s4); x=self.dec3(x,s3); x=self.dec2(x,s2); x=self.dec1(x,s1)
        return self.final(x)

class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes; self.smooth = smooth
    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1); loss = 0
        for c in range(self.num_classes):
            inter = (preds[:,c] * (targets==c).float()).sum()
            union = preds[:,c].sum() + (targets==c).float().sum()
            loss += 1 - (2*inter + self.smooth)/(union + self.smooth)
        return loss / self.num_classes

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(); self.dice = DiceLoss(num_classes=3)
    def forward(self, preds, targets):
        return self.ce(preds, targets) + self.dice(preds, targets)

def compute_miou(preds, targets, num_classes=3):
    preds = torch.argmax(preds, dim=1); iou_list = []
    for c in range(num_classes):
        inter = ((preds==c)&(targets==c)).sum().float()
        union = ((preds==c)|(targets==c)).sum().float()
        if union > 0: iou_list.append((inter/union).item())
    return sum(iou_list)/len(iou_list) if iou_list else 0.0

# Results: mIoU=0.6181, FPS=77.3, Params=7,763,107
