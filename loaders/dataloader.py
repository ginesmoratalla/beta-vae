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
    print("[DATA] Loading train dataset")
    train_dataset = datasets.MNIST(root=DATASET_PATH, train=True, transform=transforms.ToTensor(), download=True)
    print("[DATA] Loading test dataset")
    test_dataset = datasets.MNIST(root=DATASET_PATH, train=False, transform=transforms.ToTensor(), download=True)
    train_size = int(len(train_dataset) * 0.6)
    val_size = len(train_dataset) - train_size

    print("[DATA] Splitting train dataset")
    train_dataset, validation_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader
