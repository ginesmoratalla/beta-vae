import torch
import torch.nn as nn

# Optimization algorithms, e.g., sgd, adam
import torch.optim as optim
import torch.nn.functional as F

from loaders.dataloader import get_data_loaders
from models.vae import VariationalAutoEncoder

device = "mps"

# --- Hyperparameters ---
BETA = 0.1
num_classes = 18
lr = 0.001
num_epochs = 20
# --- Hyperparameters ---


# --- Hyperparameters ---
train_loader, test_loader = get_data_loaders(batch_size=150)
model = VariationalAutoEncoder(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)


def train_model():

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        print(f"[TRAINING] epoch {epoch}")
        for i, (data, targets) in enumerate(train_loader):

            data = data.float().to(device)

            # data = data.to(device)
            targets = targets.to(device)

            # forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # gradient descent
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss {loss.item():.4f}")

    print("Model finished training.\n")


def check_accuracy(loader, model, training):

    if training:
        print("[EVAL] Checking accuracy on training dataset")
    else:
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
    print("[PRE-TRAIN] Training model...")
    train_model()
    check_accuracy(train_loader, model, training=True)
