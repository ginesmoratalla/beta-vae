import torch
import torch.nn as nn

from loaders.dataloader import get_mnist_loaders

device = "cuda"
SAMPLES_PER_CLASS=8
BATCH_SIZE=64

# --- Data & Architecture ---
model = torch.load('runs/vanilla-vae/')
loader, _ = get_mnist_loaders(batch_size=BATCH_SIZE) 
# --- Data & Architecture ---


@torch.no_grad()
def inference():

    print("[EVAL] Checking accuracy on test dataset")

    model.eval()
    num_correct = 0
    num_samples = 0

    # Get mean and std of every class in the dataset
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
    inference()
