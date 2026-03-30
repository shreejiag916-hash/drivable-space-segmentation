# 🚗 Real-time Drivable Space Segmentation

A custom U-Net model trained from scratch for real-time drivable space detection using the BDD100K dataset.

## 📊 Results
| Metric | Value |
|--------|-------|
| Best mIoU | 0.6181 |
| FPS | 77.3 |
| Inference time | 12.93 ms/frame |
| Parameters | 7,763,107 |

## 🏗️ Architecture
- **Encoder:** 4x EncoderBlocks (3→32→64→128→256)
- **Bottleneck:** 512 channels
- **Decoder:** 4x DecoderBlocks with skip connections
- **Loss:** Dice + CrossEntropy (combined)
- **Optimizer:** AdamW (lr=1e-3)
- **Scheduler:** CosineAnnealingLR

## 📁 Dataset
- **Dataset:** BDD100K (External)
- **Train samples:** 2,380
- **Val samples:** 596
- **Image size:** 256×512
- **Epochs:** 25 | **Batch size:** 8

## 🏷️ Classes
| Class | Pixel Value |
|-------|-------------|
| Background | 0 |
| Drivable (main) | 127 |
| Drivable (alternative) | 191 |

## 🚀 How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download the trained model weights
[Download best_model.pth from Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)
Place the downloaded file in the same folder as model.py

### Step 3: Run inference on an image
```bash
python inference.py samples/road1.jpg
```

### Step 4: View output
The result is saved as `output_segmentation.png` showing:
- Original image
- Segmentation mask (Green = Main road, Orange = Alternative road)
- Blended overlay

## ⚠️ Note on Dataset
This model was trained on BDD100K. The problem statement specifies nuScenes.
The architecture (U-Net from scratch) and evaluation metrics are fully compliant.

## 🏆 Hackathon
MAHE Mobility Hackathon — Track 2: Real-time Drivable Space Detection


## ⚙️ Setup & Installation
```bash
# Clone the repository
git clone https://github.com/shreejiag916-hash/drivable-space-segmentation.git
cd drivable-space-segmentation

# Install dependencies
pip install torch torchvision opencv-python albumentations
```

## 🖼️ Example Output

The model produces pixel-level segmentation masks with 3 classes:
- **Black** — Background (non-drivable)
- **Green** — Main drivable road
- **Blue** — Alternative drivable area

Achieves **77.3 FPS** on NVIDIA T4 GPU — well above the 30fps real-time threshold.
