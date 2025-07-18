from tqdm import tqdm
import torch
import torch.nn as nn
import os

from torchvision.utils import make_grid
from loaders.dataloader import get_mnist_loaders
from models.vae import VariationalAutoEncoder
from utils.visualization import gif_from_tensors

device = "cuda"
Z_DIM = 60
NUM_CLASSES = 10 
SAMPLES_PER_CLASS = 10
CHANNELS = 1 
IMG_SIZE = 28 
IMAGE_FLAT_DIM = 64*4*4
BATCH_SIZE=64

INFERENCE_NAME="beta_1_third.png"
CLASS_INFERENCE_NAME="beta_1_third_samples_per_class.png"

# --- Data & Architecture ---
root_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../',
)
model_path = os.path.join(
    root_path,
    'runs/beta-vae/beta_1_third/model.pt'
)
model_weights = torch.load(model_path, weights_only=False, map_location=device)
model = VariationalAutoEncoder(
    in_channels=1,
    z_dim=Z_DIM,
    flat_dim=IMAGE_FLAT_DIM
).to(device)
model.load_state_dict(model_weights)
loader, _ = get_mnist_loaders(batch_size=BATCH_SIZE) 


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
        gif_name=INFERENCE_NAME,
    )

    model.train()

@torch.no_grad()
def inference_per_class():

    print("[EVAL] Sampling from the prior per class")
    model.eval()

    per_class_dictionary = {i:[] for i in range(NUM_CLASSES)}
    per_class_dictionary_z = {i:(torch.empty(0), torch.empty(0)) for i in range(NUM_CLASSES)}

    # Iterate through the loader
    loop = tqdm(enumerate(loader))
    for _, (x, y) in loop:
        for label in per_class_dictionary.keys():
            match = x[y == label] 
            if match.shape[0] == 0:
                continue
            per_class_dictionary[label].append(match)


    # Get mean and standard deviation for each class independently
    for label in per_class_dictionary.keys():
        label_mu = torch.empty(0, Z_DIM).to(device)
        label_sigma = torch.empty(0, Z_DIM).to(device)
        for batch in per_class_dictionary[label]:
            try:
                mu, sigma, _, _ = model(batch.to(device))
                label_mu = torch.cat((mu, label_mu), dim=0)
                label_sigma = torch.cat((sigma, label_sigma), dim=0)
            except Exception:
                print(f'[ERROR] Batch has unexpected size {batch.shape}')
                exit(0)


        print(f'[EVAL] Class {label}: Z dimension parameter shape {label_sigma.shape}')
        per_class_dictionary_z[label] = (
            torch.mean(label_mu, dim=0),
            torch.mean(label_sigma, dim=0)
        )

    del per_class_dictionary
    unorganized_batch = torch.empty(0, SAMPLES_PER_CLASS, CHANNELS, IMG_SIZE, IMG_SIZE).to(device)
    for label in per_class_dictionary_z.keys():

        mu = per_class_dictionary_z[label][0]
        sigma = per_class_dictionary_z[label][1]

        z_samples = torch.normal(
            mean=mu.expand(SAMPLES_PER_CLASS, -1),
            std=sigma.expand(SAMPLES_PER_CLASS, -1)
        )

        reconstructed_batch, _, _ = model.decode(z_samples)
        unorganized_batch = torch.cat((unorganized_batch, reconstructed_batch.unsqueeze(0)), dim=0)

    # [Class=10, B=5, C=1, H=28, W=28]
    organized_batch = unorganized_batch.transpose(0, 1).flatten(0, 1)
    print("[UNORGANIZED FINAL BATCH OF IMAGES]: ", unorganized_batch.shape)
    print("[ORGANIZED FINAL BATCH OF IMAGES]: ", organized_batch.shape)

    img_grid = make_grid(organized_batch, nrow=10)
    gif_from_tensors(
        img_sequence_list=[img_grid],
        path=root_path,
        frame_duration=0.5,
        gif_name=CLASS_INFERENCE_NAME,
    )
    model.train()


if __name__ == "__main__":
    # inference()
    inference_per_class()
