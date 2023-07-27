import os
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def get_grid_images(images, **kwargs):
    return torchvision.utils.make_grid(images, nrow=int(np.sqrt(images.shape[0])), **kwargs)

def plot_images(grid):
    plt.figure(figsize=(128, 128))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.show()

def save_images(grid, path):
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
    
def setup_logging(run_name, model_save_dir, result_save_dir):
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(result_save_dir, exist_ok=True)