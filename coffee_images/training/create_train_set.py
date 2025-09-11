import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
# Paths
MASKS_DIR = ""
RAW_IMG_DIR = ""
    


# Helper to get original image filename from mask filename
def get_original_filename(mask_filename):
    # Assumes mask filename is 'mask_' + original filename
    if mask_filename.startswith('mask_'):
        return mask_filename[len('mask_'):]
    return mask_filename

def main(n: int = None, datadir = None):
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
        os.makedirs(f'{datadir}/features/', exist_ok=True)
        cv2.imwrite(f'{datadir}/features/{orig_file}', img)
        os.makedirs(f'{datadir}/labels/', exist_ok=True)
        cv2.imwrite(f'{datadir}/labels/{orig_file}_mask', mask)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, help='Number of images to process')
    parser.add_argument('--dir', type=str, default="./data", help='Dir for data')
    parser.add_argument('--masks', type=str, default="manual_masks", help='Dir for masks')
    parser.add_argument('--raw_img', type=str, default="raw_img", help='Dir for raw images')
    args = parser.parse_args()
    
    MASKS_DIR = args.masks
    RAW_IMG_DIR = args.raw_img
    main(n=args.samples, datadir=args.dir)
