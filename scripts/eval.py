from os.path import join
from posixpath import abspath
from sys import path
from numpy import mean, rec
import torch
import torch.nn as nn
import os

from torchvision.utils import make_grid
from loaders.dataloader import get_mnist_loaders
from models.vae import VariationalAutoEncoder
from utils.visualization import gif_from_tensors

device = "mps"
Z_DIM = 70
BATCH_SIZE=64

# --- Data & Architecture ---
root_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../',
)
model_path = os.path.join(
    root_path,
    'runs/vanilla-vae/15_epochs/model.pt'
)
model = torch.load(model_path, weights_only=False, map_location=device).to(device)


@torch.no_grad()
def inference():
    """
    Decoder generates images by sampling z ~ N(0, 1)
    """
    print("[EVAL] Sampling from the prior")
    model.eval()
    mu = torch.zeros(BATCH_SIZE, Z_DIM)
    sigma = torch.ones(BATCH_SIZE, Z_DIM)
    z_samples = torch.normal(mean=mu, std=sigma).to(device)
    reconstructed_batch, _, _ = model.decode(z_samples)
    img_grid = make_grid(reconstructed_batch, nrow=8)

    gif_from_tensors(
        img_sequence_list=[img_grid],
        path=root_path,
        frame_duration=0.5,
        gif_name='samples.png',
    )

    model.train()

@torch.no_grad()
def inference_per_class():


    print("[EVAL] Sampling from the prior")
    model.eval()
    mu = torch.zeros(BATCH_SIZE, Z_DIM)
    sigma = torch.ones(BATCH_SIZE, Z_DIM)
    z_samples = torch.normal(mean=mu, std=sigma).to(device)
    reconstructed_batch, _, _ = model.decode(z_samples)
    img_grid = make_grid(reconstructed_batch, nrow=8)

    gif_from_tensors(
        img_sequence_list=[img_grid],
        path=root_path,
        frame_duration=0.5,
        gif_name='samples.png',
    )

    model.train()


if __name__ == "__main__":
    inference()
