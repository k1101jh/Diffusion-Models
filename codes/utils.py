import os
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt


def get_grid_images(images, **kwargs):
    return torchvision.utils.make_grid(images, **kwargs)
    # return grid.permute(1, 2, 0).to('cpu').numpy()


# def plot_images(images):
#     plt.figure(figsize=(32, 32))
#     plt.imshow(torch.cat([
#         torch.cat([i for i in images.cpu()], dim=-1),
#     ], dim=-2).permute(1, 2, 0).cpu())
#     plt.show()

def plot_images(grid):
    plt.figure(figsize=(32, 32))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in images.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(grid, path, **kwargs):
    # grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)