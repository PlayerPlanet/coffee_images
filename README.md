# install with Poetry
poetry install 

# Running the image labeling:
poetry run label --samples 50
Set --samples to the number of images you want to label

* press q to move to next image
* left-click to set points for polygons
* press n for new polygon
* press m to show generated mask

# Training:

poetry remove torch torchvision
poetry add --source torch-gpu torch@2.8.0 torchvision@0.23.0
# or for XPU
poetry add --source intel-xpu torch@2.8.10+xpu torchvision@0.23.0

You can add this to pyproject.toml

[tool.poetry.group.xpu.dependencies]
torch = { version = "2.8.10+xpu", source = "intel-xpu" }
torchvision = { version = "0.23.0", source = "intel-xpu" }
[[tool.poetry.source]]
name = "intel-xpu"
url = "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
priority = "supplemental"
