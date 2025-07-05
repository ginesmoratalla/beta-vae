from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.datasets as datasets

import numpy as np
import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATASET_PATH = os.path.join(ROOT_DIR, 'dataset/')


def get_mnist_loaders(batch_size):
    """
    dataset preparation

    NOTE TO SELF: transforms.ToTensor also
    normalizes the pixel values so they can
    be displayed and so that sigmoid works
    """
    print("==" * 20)
    print("[DATA] Loading train dataset")
    train_dataset = datasets.MNIST(root=DATASET_PATH, train=True, transform=transforms.ToTensor(), download=True)
    print("[DATA] Loading test dataset")
    val_dataset = datasets.MNIST(root=DATASET_PATH, train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader
