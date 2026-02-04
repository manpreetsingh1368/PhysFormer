# physformer/utils.py
import torch
from torchvision.utils import save_image

def save_patch_image(tensor, path="output.png"):
    save_image(tensor.cpu(), path)
    print(f"Image saved: {path}")
