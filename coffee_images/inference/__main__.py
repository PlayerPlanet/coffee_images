import torch
import argparse
import os
import gc
import psutil
from typing import List, Tuple
from tqdm import tqdm
from coffee_images.inference.tools import (
    load_model, inference_batch, get_image_filenames, load_image_batch,
    save_predictions, compute_masked_images
)

def main():
    """Memory-optimized main function with streaming processing"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="coffee_images/inference/segnet_weights.pth.tar")
    parser.add_argument('--image_dir', type=str, default="raw_img/")
    parser.add_argument('--n_img', type=str, default="all")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference")
    parser.add_argument('--save_interval', type=int, default=100, help="Save results every N images")
    parser.add_argument('--num_workers', type=int, default=20, help="Number of CPU cores to use")
    parser.add_argument('--target_size', type=int, nargs=2, default=None, help="Target size [height, width] for resizing images to uniform dimensions")
    args = parser.parse_args()
    
    # Set torch threads for CPU optimization
    torch.set_num_threads(args.num_workers)
    
    print(f"Loading model from {args.model_dir}...")
    model = load_model(path=args.model_dir)
    
    n_img = None if args.n_img in [None, "all"] else int(args.n_img)
    print(f"Getting image filenames from {args.image_dir}...")
    all_filenames = get_image_filenames(image_dir=args.image_dir, n=n_img)
    
    if not all_filenames:
        print("No images found!")
        return
    
    print(f"Found {len(all_filenames)} images to process")
    print(f"Processing in batches of {args.batch_size} with streaming approach")
    if args.target_size:
        print(f"Target image size: {args.target_size[0]}x{args.target_size[1]}")
    else:
        print("No resizing applied - images must have uniform dimensions")
    
    # Memory monitoring
    def print_memory_usage():
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"Current memory usage: {memory_mb:.1f} MB")
    
    # Storage for periodic saving
    batch_masks = []
    batch_results = []
    processed_count = 0
    
    print_memory_usage()
    
    # Process images in streaming batches
    with tqdm(total=len(all_filenames), desc="Processing images") as pbar:
        for i in range(0, len(all_filenames), args.batch_size):
            batch_end = min(i + args.batch_size, len(all_filenames))
            batch_filenames = all_filenames[i:batch_end]
            
            try:
                # Load only current batch into memory
                batch_images, valid_filenames = load_image_batch(
                    args.image_dir, 
                    batch_filenames, 
                    target_size=tuple(args.target_size) if args.target_size else None
                )
                
                if not batch_images:
                    pbar.update(len(batch_filenames))
                    continue
                
                # Run inference on current batch
                masks = inference_batch(batch_images, model, len(batch_images))
                
                # Store results for periodic saving
                for fname, img, mask in zip(valid_filenames, batch_images, masks):
                    batch_masks.append((fname, mask))
                    batch_results.append((fname, img, mask))
                    processed_count += 1
                
                # Clear batch from memory immediately
                del batch_images, masks
                gc.collect()
                
                # Periodic saving and memory cleanup
                if processed_count % args.save_interval == 0:
                    print(f"\nSaving results for images {processed_count - len(batch_masks) + 1} to {processed_count}")
                    save_predictions(batch_masks)
                    compute_masked_images(batch_results)
                    
                    # Clear saved results from memory
                    batch_masks.clear()
                    batch_results.clear()
                    gc.collect()
                    
                    print_memory_usage()
                
                pbar.update(len(batch_filenames))
                
            except Exception as e:
                print(f"Error processing batch {i//args.batch_size + 1}: {e}")
                # Clear memory and continue
                gc.collect()
                pbar.update(len(batch_filenames))
                continue
    
    # Save any remaining results
    if batch_masks:
        print(f"\nSaving final batch of {len(batch_masks)} results...")
        save_predictions(batch_masks)
        compute_masked_images(batch_results)
    
    print(f"\nCompleted processing {processed_count} images!")
    print(f"Results saved in 'pred_masks' and 'masked_images' directories")
    print_memory_usage()

if __name__ == "__main__":
    main()
    
