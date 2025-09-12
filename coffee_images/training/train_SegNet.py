import os
from coffee_images.training.SegNet import SegNet, Train
import torch
import torchvision.transforms as transforms
import numpy as np
from coffee import CoffeeDataset
import argparse

def main(n: int):
    datadir = 'coffee_images/training'  # Change to your data directory
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = CoffeeDataset(datadir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4)

    # Import your SegNet model here
    # from your_model_file import SegNet
    model = SegNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    Train.Train(trainloader=trainloader,epochs=n,optimizer=optimizer,model=model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of images to process')
    args = parser.parse_args()
    main(n=args.epochs)

