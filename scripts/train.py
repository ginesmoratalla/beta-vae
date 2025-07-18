import sys
from matplotlib.pyplot import cla
from rich import print
import numpy as np
from torch._prims_common import Dim
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

# Project imports
from loaders.dataloader import get_mnist_loaders
from models.vae import VariationalAutoEncoder
from utils.model_handler import save_model
from utils.visualization import gif_from_tensors, PCA

# --- Hyperparameters ---
BETA = 1
Z_DIM = 60
IMAGE_FLAT_DIM = 64*4*4
LR = 3e-4
NUM_EPOCHS = 20
BATCH_SIZE = 64
CHANNELS = 1
IMG_SIZE = 28 
device = "cuda"

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
    global BETA
    conv_layer_timestamps = get_conv_dict()
    train_reconstruction_evolution_gif = []
    validation_reconstruction_evolution_gif = []

    fixed_train_batch = next(iter(train_loader))  # For reconstruction recording
    fixed_val_batch = next(iter(val_loader))      # For reconstruction recording

    pca_y = torch.empty(0)
    pca_x = torch.empty(0, CHANNELS, IMG_SIZE, IMG_SIZE)
    for i, (x, y) in enumerate(val_loader):
        if i == 7:
            break
        pca_y = torch.cat((pca_y, y))
        pca_x = torch.cat((pca_x, x), dim=0)

    pca_batch = (pca_x.to(device), pca_y.to(device))

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
            mu, sigma, x_hat, conv_layers = model(x)
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

                train_rec, conv_layer_output = log_tensorboard(
                    writer,
                    fixed_train_batch,
                    counter,
                    conv_layers=conv_layers,
                    validation=False
                )
                validation_rec, _ = log_tensorboard(
                    writer,
                    fixed_val_batch,
                    counter,
                    conv_layers=(),
                    validation=True
                )
                train_reconstruction_evolution_gif.append(train_rec)
                validation_reconstruction_evolution_gif.append(validation_rec)
                for i, layer_ts in enumerate(conv_layer_output):
                    conv_layer_timestamps[i].append(layer_ts)

            if counter % 1000 == 0:
                print(
                        f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
                        f"Step [{i+1}/{n_total_steps}], "
                        f"Loss (per sample) {loss.item():.4f}, "
                        f"Reconstruction Loss BCE (per batch) {reconstruction_loss:.4f}, "
                        f"KL_Div: (per batch) {kl_div:.4f}"
                      )

            counter += 1

        if not epoch < 10:
            BETA += 0.3

        # TensorBoard Checkpoint
        writer.add_scalar('training loss mean (per batch)', np.mean(epoch_loss), epoch)
        writer.add_scalar('training loss std (per batch)', np.std(epoch_loss), epoch)
        writer.add_scalar('KL divergence mean (per batch)', np.mean(epoch_kl_div), epoch)
        writer.add_scalar('KL divergence std (per batch)', np.std(epoch_kl_div), epoch)
        writer.add_scalar('Reconstruction loss mean (per batch)', np.mean(epoch_reconstruction_loss), epoch)
        writer.add_scalar('Reconstruction loss std (per batch)', np.std(epoch_reconstruction_loss), epoch)

        classes = torch.rand(10)
        PCA(model, pca_batch, epoch=epoch, path=run_path)

    writer.close()
    print("Model finished training.\nLoging metrics...")
    metrics_to_save = []
    training_reconstruction = {
        'image_sequence': train_reconstruction_evolution_gif,
        'frame_duration': 0.7,
        'out_file_name': 'training_reconstruction.gif',
    }
    validation_reconstruction = {
        'image_sequence': validation_reconstruction_evolution_gif,
        'frame_duration': 0.7,
        'out_file_name': 'validation_reconstruction.gif',
    }
    metrics_to_save.append(validation_reconstruction)
    metrics_to_save.append(training_reconstruction)
    for i in conv_layer_timestamps.keys():
        metrics_to_save.append(
            {
                'image_sequence': conv_layer_timestamps[i],
                'frame_duration': 0.9,
                'out_file_name': f'conv{i+1}_outputs.gif',
            }
        )
    post_training(run_path, metrics_to_save)


@torch.no_grad()
def get_conv_dict() -> dict:
    """
    Creates dictionary that
    gets outputs of all the convolutional layers
    in the VAE
    """
    cdict = {}
    model.eval()
    i = 0
    for name, _ in model.named_modules():
        if "conv" in name:
            cdict[i] = []
            i += 1
    model.train()
    return cdict


@torch.no_grad()
def post_training(path, metric_list):
    for metric in metric_list:
        gif_from_tensors(
            img_sequence_list=metric['image_sequence'],
            path=path,
            frame_duration=metric['frame_duration'],
            gif_name=metric['out_file_name'],
        )


@torch.no_grad()
def log_tensorboard(
        writer: SummaryWriter,
        batch: torch.Tensor,
        step: int,
        conv_layers: tuple[torch.Tensor, ...],
        validation=True,
):
    mode = "(Validation)" if validation else "(Training)"
    model.eval()
    x, _ = batch
    x = x[:32].to(device)  # [BATCH, 1, 28, 28]
    _, _, reconstructed_batch, _ = model(x)
    reconstructed_batch = reconstructed_batch[:32]
    stacked_grid = torch.stack([x, reconstructed_batch], dim=1).flatten(0, 1)
    img_grid = make_grid(stacked_grid, nrow=8)
    writer.add_image(f'Image Reconstructions {mode}', img_grid, step)

    # Store conv layer output as image grid
    # Layer dims: [batch, filters, kernel, kernel]
    conv_layers_gird = []
    for i, layer in enumerate(conv_layers):
        layer_stacked_grid = []
        # Iter through every filter in a layer
        for j in range(layer.shape[1]):
            filter = layer[:, j, :, :]
            filter = torch.mean(filter, dim=0, keepdim=True)  # Mean over the whole batch
            layer_stacked_grid.append(filter)

        layer_stacked_grid = torch.stack(layer_stacked_grid, dim=0)
        if i+1 == len(conv_layers):
            layer_stacked_grid = model.sigmoid(layer_stacked_grid)

        layer_img_gird = make_grid(layer_stacked_grid, nrow=8, padding=1, pad_value=255)
        conv_layers_gird.append(layer_img_gird)
        writer.add_image(f'Conv layer {i+1} outputs', layer_img_gird, step)

    model.train()
    return img_grid.detach().cpu(), conv_layers_gird


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
    RUN_PATH = sys.argv[1]
    print("[PRE-TRAIN] Training model...")
    print("==" * 20)
    train_model(run_path=RUN_PATH)
    save_model(model, RUN_PATH)
