import torch
import torch.nn.functional as F
import os
import cv2
from coffee_images.training.SegNet import SegNet
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegNet(out_chn=2)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)
    return model

def inference_batch(images: List[torch.Tensor], model: SegNet, batch_size: int = 8) -> List[torch.Tensor]:
    """Process images in batches for better efficiency"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    results = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        
        # Prepare batch tensor
        batch_tensors = []
        for img in batch_images:
            if img.dim() == 3:  # (H, W, C)
                img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            batch_tensors.append(img)
        
        # Stack into batch
        batch = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            output = model(batch)
            batch_results = output.argmax(dim=1)
            results.extend([mask.cpu() for mask in batch_results])
    
    return results

def inference(image: torch.Tensor, model: SegNet):
    """Single image inference - kept for backward compatibility"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.permute(2,0,1)  # (H, W, C) -> (C, H, W)
    image = image.unsqueeze(0)     # (C, H, W) -> (1, C, H, W)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        return output.argmax(dim=1).cpu()
    
def get_image_filenames(image_dir: str, n: int = None):
    """Get list of image filenames without loading images"""
    fnames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    if n:
        fnames = fnames[:n]
    return fnames

def load_single_image(image_dir: str, filename: str, target_size: Optional[Tuple[int, int]] = None):
    """Load a single image on demand with optional resizing
    
    Args:
        image_dir: Directory containing the image
        filename: Name of the image file
        target_size: Optional tuple (height, width) for resizing
    
    Returns:
        torch.Tensor: Image tensor with shape (H, W, C) or None if loading failed
    """
    file_path = os.path.join(image_dir, filename)
    if not os.path.exists(file_path):
        print(f"Original image not found for: {file_path}")
        return None
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Failed to load image: {file_path}")
            return None
        
        # Convert to tensor first
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0
        
        # Resize if target_size is specified
        if target_size is not None:
            # Convert to CHW format for interpolation
            img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)  # CHW -> BCHW
            # Resize using bilinear interpolation
            img_tensor = F.interpolate(
                img_tensor, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            # Remove batch dimension and convert back to HWC
            img_tensor = img_tensor.squeeze(0).permute(1, 2, 0)  # BCHW -> CHW -> HWC
        
        return img_tensor
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_image_batch(image_dir: str, filenames: List[str], target_size: Optional[Tuple[int, int]] = None):
    """Load a batch of images efficiently with optional uniform resizing
    
    Args:
        image_dir: Directory containing images
        filenames: List of image filenames to load
        target_size: Optional tuple (height, width) for resizing all images to uniform size
    
    Returns:
        Tuple[List[torch.Tensor], List[str]]: (loaded_images, valid_filenames)
    """
    images = []
    valid_filenames = []
    
    for filename in filenames:
        img = load_single_image(image_dir, filename, target_size)
        if img is not None:
            images.append(img)
            valid_filenames.append(filename)
    
    return images, valid_filenames

def save_predictions(masks: List[Tuple[str, torch.Tensor]], output_dir: str = "pred_masks"):
    """Optimized batch saving with parallel I/O"""
    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    def save_single_mask(name_mask_tuple):
        name, mask = name_mask_tuple
        try:
            mask_np = mask.squeeze().cpu().numpy()
            if mask_np.ndim > 2:
                mask_np = mask_np[0]  # take first channel if needed
            # Convert mask to uint8
            mask_np = (mask_np > 0.5).astype('uint8') * 255
            file_path = os.path.join(output_dir, "mask_" + name)
            cv2.imwrite(file_path, mask_np)
            return True
        except Exception as e:
            print(f"Error saving mask for {name}: {e}")
            return False

    # Use ThreadPoolExecutor for parallel I/O
    with ThreadPoolExecutor(max_workers=min(8, len(masks))) as executor:
        results = list(tqdm(executor.map(save_single_mask, masks), 
                           desc="Saving predictions", total=len(masks)))
    
    successful_saves = sum(results)
    print(f"Successfully saved {successful_saves}/{len(masks)} predictions")

def compute_masked_images(imgs: List[Tuple[str, torch.Tensor, torch.Tensor]], output_dir: str = "masked_images"):
    """Optimized batch computation with parallel processing"""
    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    def compute_single_masked_image(img_tuple):
        name, img, mask = img_tuple
        try:
            file_path = os.path.join(output_dir, "masked_" + name)
            mask_np = mask.squeeze().cpu().numpy()
            # Ensure mask is 2D
            if mask_np.ndim > 2:
                mask_np = mask_np[0]  # take first channel if needed
            mask_bin = (mask_np > 0.5).astype('uint8')
            # Convert img to numpy uint8 if it's a tensor
            if isinstance(img, torch.Tensor):
                img_np = (img.cpu().numpy() * 255).astype('uint8')
            else:
                img_np = img
            # Resize if needed
            if img_np.shape[:2] != mask_bin.shape:
                mask_bin = cv2.resize(mask_bin, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Generate Gaussian noise for masked regions
            noise = np.random.normal(loc=127, scale=40, size=img_np.shape).astype('uint8')
            # Combine image and noise using mask
            masked_img = img_np.copy()
            if img_np.ndim == 3:
                for c in range(img_np.shape[2]):
                    masked_img[..., c] = np.where(mask_bin == 0, noise[..., c], img_np[..., c])
            else:
                masked_img = np.where(mask_bin == 1, noise, img_np)
            cv2.imwrite(file_path, masked_img)
            return True
        except Exception as e:
            print(f"Error computing masked image for {name}: {e}")
            return False

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(8, len(imgs))) as executor:
        results = list(tqdm(executor.map(compute_single_masked_image, imgs), 
                           desc="Computing masked images", total=len(imgs)))
    
    successful_computes = sum(results)
    print(f"Successfully computed {successful_computes}/{len(imgs)} masked images")



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
