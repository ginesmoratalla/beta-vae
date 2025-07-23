from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import transforms
import torchvision.datasets as datasets

import numpy as np
import pandas as pd
import torch
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
    print("==" * 20)
    print("[DATA] Loading train dataset (CelebA)")
    print("[DATA] Loading test dataset (CelebA)")
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../',)
    df = pd.read_csv(os.path.join(root_path,'dataset/celeba/list_attr_celeba.csv'))
    df.drop(columns=['image_id'], inplace=True)
    df.reset_index(inplace=True)
    y = df['index']
    celeba_path = os.path.join(DATASET_PATH, 'celeba')
    celeba_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image_dataset = datasets.ImageFolder(root=celeba_path, transform=celeba_transforms)
    celeba_dataset = CelebA(image_dataset, y)

    train_length = int(np.ceil(len(celeba_dataset) * 0.8))
    val_length = int(len(celeba_dataset) - train_length)
    train, val = random_split(celeba_dataset, [train_length, val_length])
    print(f'[DATA] Training set size: {train.__len__()}')
    print(f'[DATA] Validation set size: {val.__len__()}')


    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=5)
    validation_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=False, num_workers=5)

    return train_loader, validation_loader


class CelebA(Dataset):

    def __init__(self, imagefolder, indices):

        self.index_map = indices.index.tolist()
        self.imagefolder = imagefolder
        self.num_samples = len(indices)

    def __getitem__(self, index):
        real_index = self.index_map[index]
        sample, _ = self.imagefolder[real_index]
        return sample, real_index

    def __len__(self):
        return self.num_samples


def get_celeba_by_type(batch_size, property='Bald'):
    """
    dataset preparation
    """
    print("==" * 20)
    print("[DATA] Loading train dataset (CelebA)")
    print("[DATA] Loading test dataset (CelebA)")

    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../',)
    df = pd.read_csv(os.path.join(root_path,'dataset/celeba/list_attr_celeba.csv'))

    properties_df = df.loc[:, property]
    matching_df = properties_df[(properties_df[property] == 1).all(axis=1)]
    not_matching_df = properties_df[(properties_df[property] == 0).all(axis=1)]

    celeba_path = os.path.join(DATASET_PATH, 'celeba')
    celeba_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    images = datasets.ImageFolder(root=celeba_path, transform=celeba_transforms)

    # DATASETS
    property_dataset = CelebA(images, matching_df)
    not_property_dataset = CelebA(images, not_matching_df)

    print(f'[DATA] Dataset matching property size: {property_dataset.__len__()}')
    print(f'[DATA] Dataset NOT matching property size: {not_property_dataset.__len__()}')

    p_dataloader = DataLoader(dataset=property_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    not_p_dataloader = DataLoader(dataset=not_property_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    return p_dataloader, not_p_dataloader 
