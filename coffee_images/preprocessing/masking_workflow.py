# Semi-Automatic Masking Workflow
# Minimize manual effort and train a segmentation model!

# 1. Imports
import cv2, json, os, random, sys
import matplotlib.pyplot as plt
from coffee_images.preprocessing.models import MultiPolygonMaskBuilder
from tqdm import tqdm
# 2. Image Loader
IMG_DIR = 'raw_img'  # Your images directory

# Filter out images that are already labeled (present in manual_masks)
img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
manual_masks_dir = 'manual_masks'

labeled_files = set()
if os.path.exists(manual_masks_dir):
    for mask_name in os.listdir(manual_masks_dir):
        if mask_name.startswith('mask_'):
            # Remove 'mask_' prefix to get original filename
            labeled_files.add(mask_name[len('mask_'):])
# Remove already labeled files from img_files
img_files = [f for f in img_files if f not in labeled_files]

# Save labeled and unlabeled filenames to JSON
all_img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
unlabeled_files = [f for f in all_img_files if f not in labeled_files]
with open('image_label_status.json', 'w', encoding='utf-8') as f:
    json.dump({
        'labeled': sorted(list(labeled_files)),
        'unlabeled': sorted(unlabeled_files)
    }, f, indent=2)

# Take a random sample of 50 files (or fewer if there are less than 50)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, default=10, help='Number of images to process')
args = parser.parse_args()

sample_size = min(args.samples, len(img_files))
sampled_files = random.sample(img_files, sample_size)

# Load the sampled images
images = [cv2.imread(os.path.join(IMG_DIR, f)) for f in sampled_files]
print(f"Loaded {len(images)} images.")

# Show the first image
plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
plt.show()

# 3. MultiPolygon Mask Builder

# 4. Manual Masking
# Draw polygons on a few images to create initial masks.
manual_masks = []
for filename, img in tqdm(zip(sampled_files, images), desc="labeling images"):
    try:
        print("opening image...")
        mask = MultiPolygonMaskBuilder(img)
        if getattr(mask, "exit_requested", False):
            break
        manual_masks.append((filename, mask.mask))
    except:
        continue

# 5. Save Manual Masks
# These become your first training labels.
if manual_masks:
    os.makedirs('manual_masks', exist_ok=True)
    for fname, mask in manual_masks:
        cv2.imwrite(f'manual_masks/mask_{fname}', mask)

print("Manual masks saved. You can now train a segmentation model using the images and masks in 'manual_masks/'.")
print("- Use a notebook or a script with PyTorch, Keras, or your favorite framework.")
print("- After initial training, predict masks on more images and correct as needed (pseudo-labeling).")
