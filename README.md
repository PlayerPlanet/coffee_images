# introduction
This project trains SegNet model for predicting coffeepots from images.

# Structure

```
coffee_images/
	data_collection/    # Scripts and tools for collecting and organizing image data
	inference/          # Inference scripts for running predictions on new images
	preprocessing/      # Preprocessing utilities for images and masks
	training/           # Training scripts and notebooks for SegNet
manual_masks/         # Manually created mask images for training/validation
raw_img/              # Raw input images for labeling and inference
.env.example             # Env variables for telegram-api
image_label_status.json  # Tracks labeling progress/statusW
pyproject.toml           # Poetry project configuration
CITATION.cff             # Citation information
README.md                # Project documentation
```

# install with Poetry
poetry install 

# Examples
## Running the image labeling:
poetry run label --samples 50
Set --samples to the number of images you want to label

* press q to move to next image
* left-click to set points for polygons
* press n for new polygon
* press m to show generated mask

## Training:
Easiest way to run training is to open coffee_images\training\train_SegNet_runner.ipynb and run all cells.
For local training you can:
poetry run coffee_images\training\train_SegNet.py
### for GPU
poetry remove torch torchvision
poetry add --source torch-gpu torch@2.8.0 torchvision@0.23.0
### for XPU
poetry add --source intel-xpu torch@2.8.10+xpu torchvision@0.23.0

You can add this to pyproject.toml

[tool.poetry.group.xpu.dependencies]
torch = { version = "2.8.10+xpu", source = "intel-xpu" }
torchvision = { version = "0.23.0", source = "intel-xpu" }
[[tool.poetry.source]]
name = "intel-xpu"
url = "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
priority = "supplemental"

## Inference:

poetry run coffee_images/inference
possible args:
- '--model_dir', type=str, default="coffee_images/inference/segnet_weights.pth.tar"
- '--image_dir', type=str, default="raw_img/"
- '--n_img', type=int, default=1

## Testing:

# Citation:

# Contact 
@PlayerPlanet
konsta.kiirikki@aalto.fi