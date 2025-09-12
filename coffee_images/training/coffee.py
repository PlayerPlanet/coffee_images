import os
import cv2
import numpy as np
import torch
from torch.utils.data import dataset

class CoffeeDataset(Dataset):
    """
    Dataset for coffee images and masks, similar to CamVid/Pavements style.
    Expects directory structure:
        datadir/
            features/
                <image files>
            labels/
                <image files>_mask
    """
    def __init__(self, datadir, transform=None):
        self.features_dir = os.path.join(datadir, 'features')
        self.labels_dir = os.path.join(datadir, 'labels')
        self.transform = transform
        self.list_img = [f for f in os.listdir(self.features_dir) if not f.startswith('.')]

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        img_name = self.list_img[idx]
        img_path = os.path.join(self.features_dir, img_name)
        mask_path = os.path.join(self.labels_dir, "mask_" + img_name)
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(mask_path)
            raise FileNotFoundError(f"Image or mask not found for {img_name}")
        target_size = (640, 480)  # (width, height)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        # Optionally convert mask to one-hot if needed (binary: 0=background, 1=object)
        mask = (mask > 127).astype(np.uint8)
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).long()
        else:
            image = torch.from_numpy(image.transpose(2,0,1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
        return image, mask

# Feature/label generation logic from create_train_set.py

def generate_features_and_labels(masks_dir, raw_img_dir, datadir, n=None):
    import random
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if n:
        mask_files = random.sample(mask_files, n)
    for mask_file in mask_files:
        mask_path = os.path.join(masks_dir, mask_file)
        orig_file = mask_file[len('mask_'):] if mask_file.startswith('mask_') else mask_file
        orig_path = os.path.join(raw_img_dir, orig_file)
        if not os.path.exists(orig_path):
            print(f"Original image not found for mask: {mask_file}")
            continue
        img = cv2.imread(orig_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"Failed to read image or mask for: {mask_file}")
            continue
        if img.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        os.makedirs(f'{datadir}/features/', exist_ok=True)
        cv2.imwrite(f'{datadir}/features/{orig_file}', img)
        os.makedirs(f'{datadir}/labels/', exist_ok=True)
        cv2.imwrite(f'{datadir}/labels/{mask_file}', mask)
