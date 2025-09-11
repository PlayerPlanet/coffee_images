import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
# Paths
MASKS_DIR = 'manual_masks'
RAW_IMG_DIR = 'raw_img'

# Helper to get original image filename from mask filename
def get_original_filename(mask_filename):
    # Assumes mask filename is 'mask_' + original filename
    if mask_filename.startswith('mask_'):
        return mask_filename[len('mask_'):]
    return mask_filename

def main(n: int = None):
    mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.jpg') or f.endswith('.png')]
    if n:
        mask_files = random.sample(mask_files, n)
        
    for mask_file in mask_files:
        mask_path = os.path.join(MASKS_DIR, mask_file)
        orig_file = get_original_filename(mask_file)
        orig_path = os.path.join(RAW_IMG_DIR, orig_file)
        if not os.path.exists(orig_path):
            print(f"Original image not found for mask: {mask_file}")
            continue
        # Read images
        img = cv2.imread(orig_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"Failed to read image or mask for: {mask_file}")
            continue
        # Resize mask if needed
        if img.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Apply mask (show masked area only)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        # Display
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title('Original')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.title('Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.title('Masked Image')
        plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.suptitle(mask_file)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10, help='Number of images to process')
    args = parser.parse_args()
    print(args.samples)
    main(n=args.samples)
