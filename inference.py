"""
================================================
 Inference Script — Real-time Drivable Space Segmentation
 Model   : Custom U-Net (trained from scratch)
 Dataset : BDD100K (External)
 MAHE Mobility Hackathon 2026 — Track 2
================================================

Usage:
    # Single image
    python inference.py --image path/to/image.jpg --weights checkpoints/best_model.pth

    # Benchmark FPS on a folder of images
    python inference.py --image_dir path/to/images/ --weights checkpoints/best_model.pth --benchmark
"""

import argparse
import os
import time
import glob

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from model import UNet

# ── Config ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Colour map: class 0 = background, 1 = main lane, 2 = alt lane
CLASS_COLORS = {
    0: (0,   0,   0),    # black  — background
    1: (0,   255, 0),    # green  — main drivable
    2: (0,   255, 255),  # cyan   — alternative drivable
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

infer_transform = A.Compose([
    A.Resize(256, 512),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(weights_path: str) -> UNet:
    model = UNet(num_classes=3).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"[Inference] Loaded weights: {weights_path}")
    return model


# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR numpy → normalised CHW tensor."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    aug = infer_transform(image=rgb)
    return aug["image"].unsqueeze(0).to(device)   # (1, C, H, W)


# ── Postprocess ───────────────────────────────────────────────────────────────
def mask_to_color(pred_mask: np.ndarray) -> np.ndarray:
    """Convert class-index mask (H, W) to RGB colour map."""
    h, w   = pred_mask.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in CLASS_COLORS.items():
        colour[pred_mask == cls] = rgb
    return colour


# ── Single-image inference ────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: UNet, image_bgr: np.ndarray):
    """
    Run inference on one BGR image.

    Returns:
        pred_mask  (H, W)   — class indices
        colour_mask(H, W, 3)— RGB visualisation
        overlay    (H, W, 3)— original image blended with mask
        latency_ms (float)  — inference time in ms
    """
    tensor = preprocess(image_bgr)

    t0    = time.perf_counter()
    logits = model(tensor)                          # (1, 3, H, W)
    latency_ms = (time.perf_counter() - t0) * 1000

    pred_mask   = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    colour_mask = mask_to_color(pred_mask)

    # Resize colour mask back to original resolution for overlay
    orig_h, orig_w = image_bgr.shape[:2]
    colour_resized  = cv2.resize(colour_mask, (orig_w, orig_h),
                                 interpolation=cv2.INTER_NEAREST)
    orig_rgb        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay         = cv2.addWeighted(orig_rgb, 0.6, colour_resized, 0.4, 0)

    return pred_mask, colour_mask, overlay, latency_ms


# ── Visualise & save ──────────────────────────────────────────────────────────
def save_result(image_bgr, colour_mask, overlay, out_path, latency_ms):
    orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_rgb.shape[:2]
    colour_resized  = cv2.resize(colour_mask, (orig_w, orig_h),
                                 interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Input Image", "Segmentation Mask", f"Overlay  ({latency_ms:.1f} ms)"]
    imgs   = [orig_rgb, colour_resized, overlay]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img); ax.set_title(title, fontsize=13); ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color=[c/255 for c in CLASS_COLORS[0]], label="Background"),
        Patch(color=[c/255 for c in CLASS_COLORS[1]], label="Main Drivable"),
        Patch(color=[c/255 for c in CLASS_COLORS[2]], label="Alt Drivable"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=11)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Inference] Saved → {out_path}  ({latency_ms:.1f} ms)")


# ── FPS benchmark ─────────────────────────────────────────────────────────────
@torch.no_grad()
def benchmark_fps(model: UNet, image_dir: str, warmup: int = 10):
    paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))[:200]
    if not paths:
        print("[Benchmark] No .jpg files found in", image_dir); return

    dummy = cv2.imread(paths[0])
    t_in  = preprocess(dummy)

    # Warmup
    for _ in range(warmup):
        model(t_in)

    # Timed run
    t0 = time.perf_counter()
    for p in paths:
        img = cv2.imread(p)
        model(preprocess(img))
    elapsed = time.perf_counter() - t0

    fps = len(paths) / elapsed
    print(f"[Benchmark] {len(paths)} frames | {elapsed:.2f}s | FPS: {fps:.1f}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   type=str, required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--image",     type=str, default=None,
                        help="Single image path")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory of images for batch / benchmark")
    parser.add_argument("--out_dir",   type=str, default="results",
                        help="Where to save output images")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run FPS benchmark on --image_dir")
    args = parser.parse_args()

    model = load_model(args.weights)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.benchmark and args.image_dir:
        benchmark_fps(model, args.image_dir)

    elif args.image:
        img = cv2.imread(args.image)
        assert img is not None, f"Could not read image: {args.image}"
        _, colour_mask, overlay, latency_ms = predict(model, img)
        stem    = os.path.splitext(os.path.basename(args.image))[0]
        out_path = os.path.join(args.out_dir, f"{stem}_result.png")
        save_result(img, colour_mask, overlay, out_path, latency_ms)

    elif args.image_dir:
        paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
        for p in paths:
            img = cv2.imread(p)
            if img is None: continue
            _, colour_mask, overlay, latency_ms = predict(model, img)
            stem     = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(args.out_dir, f"{stem}_result.png")
            save_result(img, colour_mask, overlay, out_path, latency_ms)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
