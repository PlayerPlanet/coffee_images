# install with Poetry
poetry install 

# Running the image labeling:
poetry run label --samples 50
Set --samples to the number of images you want to label

* press q to move to next image
* left-click to set points for polygons
* press n for new polygon
* press m to show generated mask

