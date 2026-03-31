import argparse
import os
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import UNet


def parse_int_list(s: str) -> List[int]:
    s = s.replace(",", " ")
    return [int(x) for x in s.split() if x.strip()]


def make_val_transform(img_h: int, img_w: int):
    return A.Compose(
        [
            A.Resize(img_h, img_w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def save_mask_png(mask: np.ndarray, out_path: str, palette=None):
    # mask: [H,W] with class indices
    if palette is None:
        # Default: class 0 black, class 1 green, class 2 blue, ...
        palette = [(0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0)]
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        rgb[mask == class_idx] = color
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, bgr)


@torch.no_grad()
def benchmark_fps(model, device, img_h: int, img_w: int, iters: int = 200):
    model.eval()
    x = torch.randn(1, 3, img_h, img_w, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = model(x)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        fps = 1000.0 / ms
        return fps, ms

    import time

    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    t1 = time.time()
    ms = ((t1 - t0) * 1000.0) / iters
    fps = 1000.0 / ms
    return fps, ms


def main():
    parser = argparse.ArgumentParser(description="Run inference for drivable space segmentation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-img-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/preds")
    parser.add_argument("--img-ext", default="jpg")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", 2))
    img_h = int(ckpt.get("img_height", 256))
    img_w = int(ckpt.get("img_width", 512))
    palette = [(0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    transform = make_val_transform(img_h, img_w)

    os.makedirs(args.output_dir, exist_ok=True)
    img_paths = []
    import glob

    img_paths = sorted(glob.glob(os.path.join(args.input_img_dir, f"*.{args.img_ext}")))
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No input images found in {args.input_img_dir}")

    fps, ms = benchmark_fps(model, device, img_h, img_w)
    print(f"[Benchmark] fps={fps:.2f} ms/frame={ms:.2f} device={device}")

    for img_path in tqdm(img_paths):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        aug = transform(image=img)
        x = aug["image"].unsqueeze(0).to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

        out_path = os.path.join(args.output_dir, f"{stem}_mask.png")
        save_mask_png(pred, out_path, palette=palette)

    print("Inference done.")


if __name__ == "__main__":
    main()

