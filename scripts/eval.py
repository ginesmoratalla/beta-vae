import torch
import torch.nn as nn

# Optimization algorithms, e.g., sgd, adam
import torch.optim as optim
import torch.nn.functional as F

from loaders.dataloader import get_data_loaders
from models.vae import VariationalAutoEncoder

device = "mps"

# --- Hyperparameters ---
num_classes = 18
# --- Hyperparameters ---

# --- Data & Architecture ---
test_loader = get_data_loaders(batch_size=150)
model = VariationalAutoEncoder(num_classes=num_classes).to(device)
# --- Data & Architecture ---


def check_accuracy(loader, model):

    print("[EVAL] Checking accuracy on test dataset")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            y = y.to(device)
            x = x.float().to(device)

            scores = model(x)
            predictions = torch.argmax(scores, dim=1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"[EVAL] Got correct {
              num_correct} / {num_samples} --> Accuracy {(float(num_correct)/float(num_samples))*100:.2f}%"
        )
        print()

    model.train()


if __name__ == "__main__":
    check_accuracy(test_loader, model)
