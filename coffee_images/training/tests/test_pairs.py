import os
from coffee import CoffeeDataset
import cv2

def check_feature_label_pairs(features_dir, labels_dir):
    feature_files = [f for f in os.listdir(features_dir) if not f.startswith('.')]
    missing_labels = []
    unreadable_files = []
    for f in feature_files:
        label_path = os.path.join(labels_dir, "mask_" + f)
        feature_path = os.path.join(features_dir, f)
        if not os.path.exists(label_path):
            missing_labels.append(f)
        else:
            img = cv2.imread(feature_path)
            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                unreadable_files.append((feature_path, label_path))
            else:
                print(f"{feature_path}: shape={None if img is None else img.shape}, dtype={None if img is None else img.dtype}")
                print(f"{label_path}: shape={None if mask is None else mask.shape}, dtype={None if mask is None else mask.dtype}")
    if missing_labels:
        print("Missing label files for:")
        for f in missing_labels:
            print(f)
    if unreadable_files:
        print("Unreadable image or mask files:")
        for feat, lab in unreadable_files:
            print(f"Feature: {feat}, Label: {lab}")
    if not missing_labels and not unreadable_files:
        print("All feature files have corresponding and readable label files.")

def test_coffee_model():
    datadir = 'coffee_images/training'
    cof = CoffeeDataset(datadir)
    print(type(cof[0]))

if __name__ == "__main__":
    check_feature_label_pairs('coffee_images/training/features', 'coffee_images/training/labels')
    test_coffee_model()
