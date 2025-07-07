import sys
from matplotlib import cm
from rich import print
import numpy as np
from torch.nn.modules import conv
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

# Project imports
from loaders.dataloader import get_mnist_loaders
from models.vae import VariationalAutoEncoder
from utils.model_handler import save_model
from utils.visualization import gif_from_tensors

# --- Hyperparameters ---
BETA = 1
Z_DIM = 70
IMAGE_FLAT_DIM = 64*4*4
LR = 3e-4
NUM_EPOCHS = 2
BATCH_SIZE = 128
device = "mps"

# --- Model Setup ---
model = VariationalAutoEncoder(
    in_channels=1,
    z_dim=Z_DIM,
    flat_dim=IMAGE_FLAT_DIM
).to(device)
train_loader, val_loader = get_mnist_loaders(batch_size=BATCH_SIZE)
loss_fn = nn.BCELoss(reduction="sum")
optimizer = optim.Adam(params=model.parameters(), lr=LR)


def train_model(run_path):
    """
    Params:

    reconstruction_loss: The sum over all images in a batch AND over each
    pixel per-image of the BCE Loss between the original x_i and its
    reconstruction x^{i}_hat.

    kl_div: The sum over all latent dimensions z_i AND over each
    batch x[0] of the KL divergence between q_theta(z|x) and p(z) = N(0, I).
    """

    conv_layers = {}  # Conv layers (to keep track of filters)
    reconstruction_evolution_gif = []
    fixed_train_batch = next(iter(train_loader))  # For reconstruction recording
    fixed_val_batch = next(iter(val_loader))      # For reconstruction recording
    writer = SummaryWriter(run_path + "/tensorboard-logs")
    writer.add_graph(model, torch.rand(BATCH_SIZE, 1, 28, 28).to(device))

    n_total_steps = len(train_loader)*NUM_EPOCHS
    counter = 0
    # print(f"[MODEL DESC]\n{summary(model, input_size=(1, 28, 28), batch_size=BATCH_SIZE, device=device)}")

    for epoch in range(NUM_EPOCHS):
        print("==" * 20)
        print(f"[TRAINING] epoch {epoch}:\n")

        # Epoch-related data
        epoch_kl_div = []
        epoch_reconstruction_loss = []
        epoch_loss = []

        loop = tqdm(enumerate(train_loader))

        for i, (x, _) in loop:

            x = x.to(device)

            # forward pass
            mu, sigma, x_hat, _ = model(x)
            reconstruction_loss = loss_fn(x_hat, x)
            kl_div = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
            loss = (reconstruction_loss + (kl_div * BETA)) / BATCH_SIZE

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # gradient descent
            optimizer.step()
            loop.set_postfix(loss=loss.item())

            # For TensorBoard
            if counter % 300 == 0:
                epoch_kl_div.append(kl_div.detach().cpu().numpy())
                epoch_reconstruction_loss.append(reconstruction_loss.detach().cpu().numpy())
                epoch_loss.append(loss.detach().cpu().numpy())
                reconstruction_evolution_gif.append(
                    log_tensorboard(
                        writer,
                        fixed_train_batch,
                        counter,
                        validation=False
                    )
                )

            if counter % 1000 == 0:
                print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{n_total_steps}], Loss {loss.item():.4f}, BCELoss {reconstruction_loss:.4f}, KL_Div: {kl_div:.4f}")
                # filter_checkpoint(counter, writer, {})

            counter += 1

        # TensorBoard Checkpoint
        writer.add_scalar('training loss mean (per batch)', np.mean(epoch_loss), epoch)
        writer.add_scalar('training loss std (per batch)', np.std(epoch_loss), epoch)
        writer.add_scalar('KL divergence mean (per batch)', np.mean(epoch_kl_div), epoch)
        writer.add_scalar('KL divergence std (per batch)', np.std(epoch_kl_div), epoch)
        writer.add_scalar('Reconstruction loss mean (per batch)', np.mean(epoch_reconstruction_loss), epoch)
        writer.add_scalar('Reconstruction loss std (per batch)', np.std(epoch_reconstruction_loss), epoch)

    writer.close()
    gif_from_tensors(
        img_sequence_list=reconstruction_evolution_gif,
        path=run_path,
        frame_duration=0.7,
        gif_name='reconstruction.gif',
    )
    print("Model finished training.\n")


@torch.no_grad()
def log_tensorboard(
        writer: SummaryWriter,
        batch: torch.Tensor,
        epoch: int,
        validation=True,
):
    mode = "(Validation)" if validation else "(Training)"

    model.eval()
    x, _ = batch
    x = x[:32].to(device)  # [BATCH, 1, 28, 28]
    _, _, reconstructed_batch, (conv_layers) = model(x)
    reconstructed_batch = reconstructed_batch[:32]
    stacked_grid = torch.stack([x, reconstructed_batch], dim=1).flatten(0, 1)
    img_grid = make_grid(stacked_grid, nrow=8)
    writer.add_image(f'Image Reconstructions {mode}', img_grid, epoch)

    if not validation:
        for i, layer in enumerate(conv_layers):  # Dimensions: [out_channels, in_channels, kernel, kernel]
            stacked_grid = []
            for j in range(layer.shape[0]):
                filter = torch.mean(layer[j, :, :, :], dim=0)
                stacked_grid.append(filter)

            stacked_grid = torch.stack(stacked_grid, dim=0).unsqueeze(1)
            img_grid = make_grid(stacked_grid, nrow=8, padding=1, pad_value=255)
            writer.add_image(f'Conv layer {i+1} outputs', img_grid, epoch)

    model.train()
    return img_grid.detach().cpu()


@torch.no_grad()
def filter_checkpoint(step: int, writer: SummaryWriter, conv_layers={}):
    """
    Track filter weights along training
    """
    model.eval()
    for i, (name, layer) in enumerate(model.named_modules()):
        if "conv" in name:
            conv_layers[i] = conv_layers.get(i, [])
            layer_weight = layer.weight
            conv_layers[i].append(layer_weight)

    for idx, layer in enumerate(conv_layers.values()):
        stacked_grid = []
        for timestamp in layer:  # Dimensions: [out_channels, in_channels, kernel, kernel]
            for i in range(timestamp.shape[0]):
                filter = torch.mean(timestamp[i, :, :, :], dim=0)
                stacked_grid.append(filter)

        stacked_grid = torch.stack(stacked_grid, dim=0).unsqueeze(1)
        img_grid = make_grid(stacked_grid, nrow=8, padding=2, pad_value=255)
        writer.add_image(f'Conv layer {idx+1} filters', img_grid, step)

    model.train()


if __name__ == "__main__":
    # filter_checkpoint()
    RUN_PATH = sys.argv[1]
    print("[PRE-TRAIN] Training model...")
    print("==" * 20)
    train_model(run_path=RUN_PATH)
    save_model(model, RUN_PATH)
