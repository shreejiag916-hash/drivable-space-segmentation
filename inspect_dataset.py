import argparse
import os
import random
from collections import Counter

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def iter_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def is_mask(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in MASK_EXTS


def load_grayscale(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img


def summarize_mask_values(mask_paths, sample_n: int, seed: int):
    rng = random.Random(seed)
    if len(mask_paths) == 0:
        return {}
    sample = mask_paths if len(mask_paths) <= sample_n else rng.sample(mask_paths, sample_n)

    value_counts = Counter()
    for p in sample:
        m = load_grayscale(p)
        # Downsample unique extraction for speed on large masks
        uniq = np.unique(m)
        for v in uniq.tolist():
            value_counts[int(v)] += 1
    return dict(sorted(value_counts.items(), key=lambda kv: kv[0]))


def main():
    ap = argparse.ArgumentParser(
        description="Inspect a downloaded dataset folder: counts images/masks, prints sample mask pixel values."
    )
    ap.add_argument("--root", required=True, help="Folder where you extracted nuScenes / masks / images.")
    ap.add_argument("--sample-masks", type=int, default=30, help="How many masks to sample for pixel values.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise NotADirectoryError(root)

    all_files = list(iter_files(root))
    image_files = [p for p in all_files if is_image(p)]
    mask_files = [p for p in all_files if is_mask(p)]

    # Heuristic: masks usually live in folders with 'mask'/'label' words
    likely_mask_files = [
        p
        for p in mask_files
        if any(k in p.lower() for k in ["mask", "masks", "label", "labels", "seg", "segmentation", "semantic"])
    ]
    if len(likely_mask_files) >= 10:
        mask_files_to_use = likely_mask_files
    else:
        mask_files_to_use = mask_files

    print(f"Root: {root}")
    print(f"Total files: {len(all_files)}")
    print(f"Image-like files: {len(image_files)}")
    print(f"Mask-like files (all): {len(mask_files)}")
    print(f"Mask-like files (likely): {len(likely_mask_files)}")
    print("")

    if len(mask_files_to_use) == 0:
        print("No mask files detected. If your masks are in a special format, tell me what you see in the folder.")
        return

    # Print a few example paths
    print("Example mask paths:")
    for p in mask_files_to_use[:5]:
        print(f"  {p}")
    print("")

    values = summarize_mask_values(mask_files_to_use, sample_n=args.sample_masks, seed=args.seed)
    print(f"Unique pixel values seen in {min(len(mask_files_to_use), args.sample_masks)} sampled masks:")
    print(values)
    print("")
    print("Next step:")
    print("- Tell me the printed pixel values dict above.")
    print("- If drivable pixels are, for example, 1 and background is 0, we will use --binary-foreground-values \"1\".")


if __name__ == "__main__":
    main()

