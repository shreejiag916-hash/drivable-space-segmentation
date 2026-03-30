# ================================================
# Inference Script - Drivable Space Segmentation
# Run: python inference.py path/to/your/image.jpg
# ================================================

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import the UNet model from model.py
from model import UNet

# Color map: Background=Black, Main Road=Green, Alt Road=Orange
COLOR_MAP = {
    0: [0, 0, 0],        # Background - Black
    1: [0, 255, 0],      # Drivable Main - Green
    2: [255, 165, 0],    # Drivable Alternative - Orange
}

def run_inference(image_path, model_path="best_model.pth"):
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' not found!")
        print("Please download best_model.pth and place it in this folder.")
        return

    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image '{image_path}' not found!")
        return

    print("Loading model...")
    model = UNet(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("Model loaded successfully!")

    # Load and prepare image
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 256))

    # Normalize (same as training)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_norm = (img_resized / 255.0 - mean) / std
    tensor = torch.tensor(img_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().numpy()

    # Create colored segmentation image
    color_mask = np.zeros((256, 512, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        color_mask[pred_mask == class_id] = color

    # Create overlay (blend original + mask)
    overlay = cv2.addWeighted(img_resized, 0.5, color_mask, 0.5, 0)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Drivable Space Segmentation - MAHE Mobility Hackathon 2026", fontsize=13)

    axes[0].imshow(img_resized)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(color_mask)
    axes[1].set_title("Segmentation Mask\n(Green=Main Road, Orange=Alt Road)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Original + Mask)")
    axes[2].axis("off")

    plt.tight_layout()
    output_filename = "output_segmentation.png"
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    print(f"\nDone! Output saved as: {output_filename}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image.jpg>")
        print("Example: python inference.py samples/road.jpg")
    else:
        run_inference(sys.argv[1])
        
