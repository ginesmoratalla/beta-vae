import torch
import torch.nn as nn
from torch.nn import KLDivLoss
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image 
from tqdm import tqdm

# Project imports
from loaders.dataloader import get_mnist_loaders
from models.vae import VariationalAutoEncoder

# --- Hyperparameters ---
BETA = 1
Z_DIM = 70
IMAGE_FLAT_DIM = 64*4*4
LR = 3e-4
NUM_EPOCHS = 20
BATCH_SIZE = 64
device = "mps"

# --- Model Setup ---
model = VariationalAutoEncoder(
    in_channels=1,
    z_dim=Z_DIM,
    flat_dim=IMAGE_FLAT_DIM
).to(device)
train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE)
loss_fn = nn.BCELoss(reduction="sum")
optimizer = optim.Adam(params=model.parameters(), lr=LR)


def train_model():

    n_total_steps = len(train_loader)*NUM_EPOCHS
    counter = 0
    for epoch in range(NUM_EPOCHS):
        print(f"[TRAINING] epoch {epoch}")
        loop = tqdm(enumerate(train_loader))
        for i, (x, _) in loop:

            x = x.to(device)

            # forward pass
            mu, sigma, x_hat = model(x)
            reconstruction_loss = loss_fn(x_hat, x) / x.shape[0]
            kl_div = -torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2) / 2
            loss = reconstruction_loss + (kl_div * BETA)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # gradient descent
            optimizer.step()
            loop.set_postfix(loss=loss.item())

            if counter % 1000 == 0:
                print("[DEBUG METRIC]: ")
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{n_total_steps}], Loss {loss.item():.4f}")

            counter += 1

    print("Model finished training.\n")


if __name__ == "__main__":
    print("[PRE-TRAIN] Training model...")
    train_model()
