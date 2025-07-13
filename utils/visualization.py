import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from torchvision import transforms

NUM_PC = 3

def gif_from_tensors(
        img_sequence_list,
        path,
        frame_duration=0.5,
        gif_name='placeholder.gif'):

    print("Generating assets...")
    transform = transforms.ToPILImage()
    store_path = path + "/res"
    os.makedirs(store_path, exist_ok=True)

    img_array = []

    for _, img in enumerate(img_sequence_list):
        img_from_tensor = transform(img)
        img_array.append(img_from_tensor)

    imageio.mimsave(
        os.path.join(store_path, gif_name),
        img_array,
        duration=frame_duration,
        loop=0
    )


def PCA(classes: torch.Tensor, mu: torch.Tensor, n: int):
    """
    Args
    classes: tensor of image input classes size [batch].
    each cell corresponds to the class of the ith sample

    mu: tensor of shape [batch, z_dimension].
    """

    print("[PCA] Calculating Principal Components for the latent dimensions")
    mean_per_dim = mu.detach().mean(dim=0)  # [z_dimension]
    centered_mu = mu - mean_per_dim         # [batch, z_dimension]
    cov_matrix = (1/n-1) * (centered_mu.T @ centered_mu)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # N.B. eigenvalues are already ordered in decreasing order
    variance = torch.sum(eigenvalues)
    eigenvalues = eigenvalues[:NUM_PC]       # Not needed for projection matrix
    eigenvectors = eigenvectors[:, :NUM_PC]  # Each column is an eigenvector

    # Explained variance
    print("==" * 20)
    print('PCA Explained Variance:')
    for i in range(eigenvalues.shape[0]):
        print(f'PC {i+1} {(eigenvalues[i]/variance)*100:.2f}%')
    print("==" * 20)

    transformed_samples = mu @ eigenvectors
