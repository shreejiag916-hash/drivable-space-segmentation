# 🚗 Real-time Drivable Space Segmentation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shreejiag916-hash/drivable-space-segmentation/blob/main/train_and_infer.ipynb)s
> **MAHE Mobility Hackathon 2026 · Track 01 (AI in Mobility) · Problem Statement 2**  
> Built in collaboration with **Harman Automotive**

A custom **U-Net model trained entirely from scratch** for real-time pixel-wise segmentation of drivable areas in complex urban driving scenarios. The model classifies every pixel into three classes — Background, Main Drivable Lane, and Alternative Drivable Lane — at well above real-time speeds.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Best mIoU** | **0.6181** |
| **Inference Speed** | **77.3 FPS** |
| Inference latency | 12.93 ms/frame |
| Parameters | 7,763,107 |
| Epochs trained | 25 |
| Training samples | 2,380 |
| Validation samples | 596 |

---

## 🏗️ Architecture — Custom U-Net

```
Input (3 × 256 × 512)
       │
  ┌────▼────┐
  │ Encoder  │  4 × EncoderBlock  (3→32→64→128→256 channels)
  │          │  Each: ConvBlock + MaxPool2d
  └────┬────┘
       │
  ┌────▼────┐
  │Bottleneck│  ConvBlock  (256 → 512 channels)
  └────┬────┘
       │
  ┌────▼────┐
  │ Decoder  │  4 × DecoderBlock  (512→256→128→64→32 channels)
  │          │  Each: ConvTranspose2d + skip-connection + ConvBlock
  └────┬────┘
       │
  ┌────▼────┐
  │  Output  │  1×1 Conv  →  3-class logit map (256 × 512)
  └─────────┘
```

**Design choices:**
- No pre-trained backbone (trained from scratch — hackathon requirement)
- `BatchNorm + ReLU` after every convolution for stable training
- Skip connections preserve fine spatial detail for sharp boundaries
- Lightweight encoder (starts at 32 channels) keeps inference fast
- **Loss:** Dice Loss + Cross-Entropy (handles class imbalance)
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR

---

## 🏷️ Segmentation Classes

| Class | Label | Pixel Value (raw) | Colour in output |
|-------|-------|-------------------|-----------------|
| Background | 0 | 0 | ⬛ Black |
| Main Drivable | 1 | 127 | 🟩 Green |
| Alt Drivable | 2 | 191 | 🟦 Cyan |

---

## 📁 Dataset

| | |
|-|--|
| **Dataset** | [BDD100K](https://bdd-data.berkeley.edu/) (External — permitted by hackathon rules) |
| **Split** | 2,380 train / 596 val |
| **Input size** | 256 × 512 px |
| **Augmentations** | HorizontalFlip, RandomBrightnessContrast, Rotate(±10°), ImageNet Normalisation |

---

## 🗂️ Repository Structure

```
drivable-space-segmentation/
├── model.py          ← U-Net architecture + loss functions + mIoU metric
├── dataset.py        ← BDD100K Dataset class + DataLoader factory
├── train.py          ← Full training loop with AMP, checkpointing, curve plots
├── inference.py      ← Single-image / batch inference + FPS benchmark
├── demo.ipynb        ← Google Colab notebook (run in one click)
├── requirements.txt
└── results/
    └── training_curves.png
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare BDD100K dataset
```
bdd100k/
├── images/
│   ├── train/   (*.jpg)
│   └── val/     (*.jpg)
└── drivable_maps/
    ├── train/   (*.png)
    └── val/     (*.png)
```

### 3. Train
```bash
python train.py --data_root /path/to/bdd100k --epochs 25 --batch_size 8
```

### 4. Run Inference
```bash
# Single image
python inference.py --weights checkpoints/best_model.pth --image sample.jpg

# Benchmark FPS
python inference.py --weights checkpoints/best_model.pth --image_dir /path/to/images --benchmark
```

### 5. Or open the Colab notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](demo.ipynb)

---

## 📈 Why These Metrics Matter

- **mIoU (0.6181):** Mean Intersection over Union — primary accuracy metric. Measures how well predicted masks overlap with ground truth across all 3 classes.
- **FPS (77.3):** Far exceeds real-time threshold (30 FPS). Critical for L4 autonomous driving perception pipelines.
- **Low parameter count (7.7M):** Compact enough to run on edge hardware inside a vehicle cockpit.

---

## 👥 Team

| Name | Role |
|------|------|
| Shreeji Agrawal | Model Architecture & Training |
| *(teammate)* | *(role)* |
| *(teammate)* | *(role)* |

---

## 🏆 Hackathon

**MAHE Mobility Challenge 2026**  
Track 01 — AI in Mobility (in collaboration with Harman Automotive)  
Problem Statement 2 — Real-time Drivable Space Segmentation
