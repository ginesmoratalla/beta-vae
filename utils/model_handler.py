import torch
import os
from datetime import datetime
import sys


def save_model(model, model_path):

    model_name = model_path + "/model.pt"
    model_path = os.path.join(model_path, model_name)
    torch.save(model, model_path)
    print("[MODEL SAVED]", model_path)


def create_run_path(model_architecture="PLACEHOLDER"):

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    date = str(datetime.now()).translate(str.maketrans({" ": "_", ".": "-", ":": "_"}))
    model_path = os.path.join(root_dir, f'runs/{model_architecture}/')
    model_path_full = os.path.join(root_dir, f'runs/{model_architecture}/', date)

    # Dir for the entire run
    if not os.path.exists(model_path_full):
        os.makedirs(model_path_full)

    print(model_path, model_path_full)


def print_stderr(msg):
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    print_stderr("==" * 20)
    print_stderr("[PIPELINE] Creating directory to store logs...")
    create_run_path("vanilla-vae")
    print_stderr("[PIPELINE] Log dir created succesfully")
