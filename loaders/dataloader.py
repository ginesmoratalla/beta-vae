import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATASET_PATH = os.path.join(ROOT_DIR, 'dataset/onepiecedata.pt')


class OnePieceDataset(Dataset):

    def __init__(self):
        xy = torch.load(DATASET_PATH)

        # Images saved in tuples, stacking adds a new dimension as 'batch'
        self.x = torch.stack([tuple[0] for tuple in xy])
        self.y = [tuple[1] for tuple in xy]
        self.num_samples = self.x.shape[0]
        # print(self.x.shape)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.num_samples


def get_data_loaders(batch_size):

    dataset = OnePieceDataset()
    X_train, X_test, y_train, y_test = train_test_split(dataset.x, dataset.y, test_size=0.2, random_state=42)
    train = TensorDataset(X_train, torch.tensor(y_train))
    test = TensorDataset(X_test, torch.tensor(y_test))

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
