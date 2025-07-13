import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from PIL import Image
from torchvision import transforms

NUM_PC = 3
MNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
COLORS = ['black', 'pink', 'blue', 'red', 'orange', 'purple', 'brown', 'yellow', 'cyan', 'magenta']

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

@torch.no_grad()
def PCA(model, pca_batch: tuple[torch.Tensor, torch.Tensor], epoch, path: str):
    """
    Args
    pca_batch: tuple of shape 
    [
        image_samples [batch_size, channels, img_dim, img_dim],
        image_labels [batch_size]
    ]
    """
    model.eval()
    mu, _, _, _ = model(pca_batch[0])

    print("[PCA] Calculating Principal Components for the latent dimensions")
    mean_per_dim = mu.detach().mean(dim=0)  # [z_dimension]
    centered_mu = mu - mean_per_dim         # [batch, z_dimension]
    cov_matrix = (1/mu.shape[0]-1) * (centered_mu.T @ centered_mu)
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

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for pair in zip(MNIST_CLASSES, COLORS):
        i = int(pair[0])
        mask = pca_batch[1] == i
        i_samples = transformed_samples[mask]
        ax.scatter(i_samples[:, 0], i_samples[:, 1], i_samples[:, 2], c=pair[1], label=pair[0])

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()

    os.makedirs(path)
    plt.savefig(path+f'pca_{epoch}.png')

    model.train()
