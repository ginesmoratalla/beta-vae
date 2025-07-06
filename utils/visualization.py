import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from torchvision import transforms


def gif_from_tensors(
        img_sequence_list,
        path,
        frame_duration=0.5,
        gif_name='placeholder.gif'):

    print("Generating reconstruction GIF...")
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
