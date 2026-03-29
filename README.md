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
```bash
pip install torch torchvision opencv-python albumentations
python model.py
```

## 🏆 Hackathon
MAHE Mobility Hackathon — Track 2: Real-time Drivable Space Detection
