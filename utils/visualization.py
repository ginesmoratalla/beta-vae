import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from torchvision import transforms


def create_reconstruction_gif(img_sequence_list, path):

    print("Generating reconstruction GIF...")
    transform = transforms.ToPILImage()
    store_path = path + "/reconstructions"
    os.makedirs(store_path, exist_ok=True)

    img_array = []

    for _, img in enumerate(img_sequence_list):
        img_from_tensor = transform(img)
        img_array.append(img_from_tensor)

    imageio.mimsave(
        os.path.join(store_path, 'reconstruction.gif'),
        img_array,
        duration=0.5,
        loop=0
    )
