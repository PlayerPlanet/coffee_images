import torch
import argparse
from coffee_images.inference.tools import load_model, inference, open_images, plot_mask

def main():
    return 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="coffee_images/inference/segnet_weights.pth.tar")
    parser.add_argument('--image_dir', type=str, default="raw_img/")
    parser.add_argument('--n_img', type=int, default=1)
    args = parser.parse_args()
    model = load_model(path=args.model_dir)
    images = open_images(image_dir=args.image_dir, n=args.n_img)
    masks = []
    for img in images:
        mask = inference(image=img, model=model)
        masks.append(mask)
    for img, mask in zip(images, masks):
        plot_mask(img=img, mask=mask)
