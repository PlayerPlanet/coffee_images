
import os
import torch
from coffee_images.inference.tools import compute_masked_images
import cv2
from tqdm import tqdm

RAW_IMG_DIR = 'raw_img'
PRED_MASKS_DIR = 'pred_masks'
OUTPUT_DIR = 'masked_images'

# Get all mask files
mask_files = [f for f in os.listdir(PRED_MASKS_DIR) if f.lower().startswith('mask_') and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

BATCH_SIZE = 16  # You can adjust this based on available memory
total = len(mask_files)
print(f"Found {total} image-mask pairs. Computing masked images in batches of {BATCH_SIZE}...")

for i in tqdm(range(0, total, BATCH_SIZE), desc="Batches"):
    batch_files = mask_files[i:i+BATCH_SIZE]
    img_mask_tuples = []
    for mask_file in batch_files:
        img_file = mask_file.replace('mask_', '')
        img_path = os.path.join(RAW_IMG_DIR, img_file)
        mask_path = os.path.join(PRED_MASKS_DIR, mask_file)
        if not os.path.exists(img_path):
            print(f"Image not found for mask: {mask_file}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask: {mask_path}")
            continue
        mask_tensor = torch.tensor(mask, dtype=torch.float32) / 255.0
        img_mask_tuples.append((img_file, img_tensor, mask_tensor))
    if img_mask_tuples:
        compute_masked_images(img_mask_tuples, output_dir=OUTPUT_DIR)
print("Done.")
