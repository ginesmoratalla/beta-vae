from torch.utils.data import DataLoader, random_split, TensorDataset
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


def get_celeba_loaders(batch_size):
    """
    dataset preparation

    NOTE TO SELF: transforms.ToTensor also
    normalizes the pixel values so they can
    be displayed and so that sigmoid works
    """

    celeba_path = os.path.join(DATASET_PATH, 'celeba')
    celeba_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    celeba_dataset = datasets.ImageFolder(root=celeba_path, transform=celeba_transforms)

    train_length = int(np.ceil(len(celeba_dataset) * 0.8))
    val_length = int(len(celeba_dataset) - train_length)
    train, val = random_split(celeba_dataset, [train_length, val_length])

    print("==" * 20)
    print("[DATA] Loading train dataset (CelebA)")
    print("[DATA] Loading test dataset (CelebA)")

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=5)
    validation_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=False, num_workers=5)

    return train_loader, validation_loader

# if __name__ == '__main__':
    # get_celeb_loaders(64)
