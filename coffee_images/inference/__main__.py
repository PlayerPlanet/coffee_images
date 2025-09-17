import torch
import argparse
from coffee_images.inference.tools import load_model, inference, open_images, save_predictions, compute_masked_images

def main():
    return 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="coffee_images/inference/segnet_weights.pth.tar")
    parser.add_argument('--image_dir', type=str, default="raw_img/")
    parser.add_argument('--n_img', type=str, default=1)
    args = parser.parse_args()
    model = load_model(path=args.model_dir)
    n_img = None if args.n_img in [None, "all"] else int(args.n_img)
    images, fnames = open_images(image_dir=args.image_dir, n=n_img)
    masks = []
    for img in images:
        mask = inference(image=img, model=model)
        masks.append(mask)
    named_masks = zip(fnames, masks)
    save_predictions(named_masks)
    imgs = list(zip(fnames, images, masks))
    compute_masked_images(imgs)
    
