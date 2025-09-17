import torch
import os
import cv2
from coffee_images.training.SegNet import SegNet
import matplotlib.pyplot as plt

def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegNet(out_chn=2)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def inference(image: torch.Tensor, model: SegNet):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.permute(2,0,1)  # (H, W, C) -> (C, H, W)
    image = image.unsqueeze(0)     # (C, H, W) -> (1, C, H, W)
    image = image.to(device)
    output = model(image)
    with torch.no_grad():
        return output.argmax(dim=1)
    
def open_images(image_dir: str, n: int = None):
    fnames =  [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if n:
        fnames = fnames[:n]
    images = []
    for file in fnames:
        file_path = os.path.join(image_dir, file)
        if not os.path.exists(file_path):
            print(f"Original image not found for: {file_path}")
            continue
        img = cv2.imread(file_path)
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0  # Convert to float32 and normalize
        images.append(img_tensor)
    return images

def plot_mask(img, mask: torch.Tensor):
    mask = mask.squeeze().cpu().numpy()
    # Ensure mask is 2D
    if mask.ndim > 2:
        mask = mask[0]  # take first channel if needed
    # Convert mask to uint8
    mask = (mask > 0.5).astype('uint8') * 255
    # Convert img to numpy uint8 if it's a tensor
    if isinstance(img, torch.Tensor):
        img = (img.cpu().numpy() * 255).astype('uint8')
    # Resize if needed
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
    plt.suptitle("generated mask")
    plt.show()
