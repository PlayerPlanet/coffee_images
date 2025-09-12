import os
from coffee_images.training.SegNet import SegNet, Train
import torch
import torchvision.transforms as transforms
import numpy as np
from coffee import CoffeeDataset
import argparse

def main(batch_size: int, datadir: str, learning_rate: int = 1e-3, epochs: int = 8):
      # Change to your data directory
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = CoffeeDataset(datadir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Import your SegNet model here
    # from your_model_file import SegNet
    model = SegNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    Train.Train(trainloader=trainloader,epochs=epochs,optimizer=optimizer,model=model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--learning_rate', type=int, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--data_dir', type=str, default="coffee_images/training", help='')

    args = parser.parse_args()
    main(batch_size=args.batch_size, datadir=args.data_dir, learning_rate=args.learning_rate, epochs=args.epochs, )

