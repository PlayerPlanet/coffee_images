import os
from coffee_images.training.SegNet import SegNet, Train
import torch
import torchvision.transforms as transforms
import numpy as np
from coffee import CoffeeDataset
import argparse

def main(batch_size: int, datadir: str, learning_rate: float = 1e-3, epochs: int = 8, checkpoint = None):
    # Change to your data directory
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Split dataset into train/val (80/20 split for demonstration)
    full_dataset = CoffeeDataset(datadir, transform=transform)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    trainset, valset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Import your SegNet model here
    model = SegNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Pass checkpoint_path if provided
    Train.Train(trainloader=trainloader, epochs=epochs, optimizer=optimizer, model=model, path=checkpoint)

    # --- Evaluation step ---
    print("\nEvaluating on validation set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            # Compute pixel accuracy
            preds = outputs.argmax(dim=1)
            correct_pixels += (preds == labels).sum().item()
            total_pixels += torch.numel(labels)
    avg_loss = total_loss / len(valset) if len(valset) > 0 else 0
    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0
    print(f"Validation Loss: {avg_loss:.4f}, Pixel Accuracy: {pixel_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--data_dir', type=str, default="coffee_images/training", help='')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (optional)')

    args = parser.parse_args()
    main(batch_size=args.batch_size, 
         datadir=args.data_dir, 
         learning_rate=args.learning_rate, 
         epochs=args.epochs,
         checkpoint=args.checkpoint)

