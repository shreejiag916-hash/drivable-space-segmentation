# 🚗 MAHE Hackathon 2026: Real-time Drivable Space Segmentation

This repo provides a **train-from-scratch** semantic segmentation pipeline for the hackathon track **“Real-time Drivable Space Segmentation”** (drivable vs non-drivable).  
The track requires **no pre-trained model weights** (our model is a small custom U-Net with randomly initialized layers).

## What you get
- `train.py`: trains the model and writes `outputs/metrics.json` (includes best validation `mIoU`) and saves `outputs/checkpoints/best.pt`
- `infer.py`: runs inference on an image folder and saves visual masks to an output folder
- Clean metrics + checkpoints so you can link working code in your submission PPT

## Dataset format (you must adapt from the provided nuScenes package)
The code expects a segmentation-style folder where **image filenames and mask filenames share the same stem**.

Example:

```text
dataset/
  train/
    images/000001.jpg
    masks/000001.png
  val/
    images/000101.jpg
    masks/000101.png
```

Mask pixel mapping:
- Default is **binary**: any mask pixel value listed in `--binary-foreground-values` becomes class `1` (drivable); everything else becomes class `0`.
- If your provided masks are already 0/1, keep the default.

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py ^
  --train-img-dir "PATH_TO_TRAIN_IMAGES" ^
  --train-mask-dir "PATH_TO_TRAIN_MASKS" ^
  --val-img-dir "PATH_TO_VAL_IMAGES" ^
  --val-mask-dir "PATH_TO_VAL_MASKS" ^
  --num-classes 2 ^
  --binary-foreground-values "1" ^
  --img-height 256 --img-width 512 ^
  --epochs 10 --batch-size 4 ^
  --save-dir outputs
```

If your drivable pixels are not `1`, set `--binary-foreground-values` to the correct values from your masks.

## Inference (for PPT screenshots)
```bash
python infer.py ^
  --checkpoint "outputs/checkpoints/best.pt" ^
  --input-img-dir "PATH_TO_VAL_IMAGES" ^
  --output-dir "outputs/preds"
```

## Hackathon notes
- The track’s primary metric is **mIoU** and it also evaluates **FPS/inference speed**.
- Code output to include in your submission:
  - `outputs/metrics.json`
  - `outputs/checkpoints/best.pt`
  - a few `outputs/preds/*.png` visualizations
