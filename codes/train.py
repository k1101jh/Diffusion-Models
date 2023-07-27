import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from accelerate import Accelerator
from ema_pytorch import EMA
import traceback

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter

from backbones.unet import UNet, UNet_conditional
from diffusions.ddpm import DDPM
from utils import get_grid_images, plot_images, save_images, setup_logging

os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def get_data(image_size, batch_size, dataset_path):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train(run_name, epochs, batch_size, image_size, dataset_path: str, lr: float, test_term: int=10, num_test_images=10, use_ema=False):
    """
    loss: MSE(noise, predicted_noise)

    Args:
        run_name (_type_): _description_
        epochs (_type_): _description_
        batch_size (_type_): _description_
        image_size (_type_): _description_
        dataset_path (str): _description_
        lr (float): _description_
    """
    accelerator = Accelerator()
    device = accelerator.device
    
    model_save_dir = "../models" + run_name
    result_save_dir = "../results" + run_name
    
    setup_logging(model_save_dir, result_save_dir)
    
    model = UNet().to(device)
    
    if use_ema:
        ema = EMA(
            model,
            beta=0.9999,
            update_after_step=100,
            update_every=10,
        )
        ema.to(device)
        
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    dataloader = get_data(image_size, batch_size, dataset_path)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    criterion = nn.MSELoss()
    diffusion = DDPM(image_size=image_size, device=device)
    writer = SummaryWriter(os.path.join("../runs", run_name))

    for epoch in range(1, epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss = 0.0
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = criterion(noise, predicted_noise)
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            epoch_loss += loss.item()
            
            if use_ema:
                ema.update()
                
        pbar.close()
        writer.add_scalar("MSE", epoch_loss, global_step=epoch)    

        if epoch % test_term == 0:
            with torch.no_grad():
                if use_ema:
                    ema_sampled_images = diffusion.sample(ema.ema_model, num_test_images)
                    ema_grid = get_grid_images(ema_sampled_images)
                    save_images(ema_grid, os.path.join(result_save_dir, f"{epoch}_ema.png"))
                    writer.add_image('sample images ema', ema_grid, global_step=epoch)
                    
                sampled_images = diffusion.sample(model, num_test_images)
                grid = get_grid_images(sampled_images)
                save_images(grid, os.path.join(result_save_dir, f"{epoch}.png"))
                writer.add_image('sample images', grid, global_step=epoch)
            
            save_dict = {
                'model': model.state_dict(),
                'ema_model': ema.state_dict() if use_ema else None,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(save_dict, os.path.join(model_save_dir, f"ckpt.pt"))
                
    # conditional
    model = UNet_conditional().to(device)
    
    if use_ema:
        ema = EMA(
            model,
            beta=0.9999,
            update_after_step=100,
            update_every=10,
        )
        ema.to(device)
        
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    dataloader = get_data(image_size, batch_size, dataset_path)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    criterion = nn.MSELoss()
    diffusion = DDPM(image_size=image_size, device=device)
    writer = SummaryWriter(os.path.join("../runs", run_name))
    
    for epoch in range(1, epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t)
            loss = criterion(noise, predicted_noise)
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            epoch_loss += loss.item()
            
            if use_ema:
                ema.update()

        pbar.close()
        writer.add_scalar("MSE", epoch_loss, global_step=epoch)    

        if epoch % test_term == 0:
            with torch.no_grad():
                labels = torch.arange(10).long().to(device)
                if use_ema:
                    ema_sampled_images = diffusion.sample(ema.ema_model, num_test_images, labels=labels)
                    ema_grid = get_grid_images(ema_sampled_images)
                    save_images(ema_grid, os.path.join(result_save_dir, f"{epoch}_ema.png"))
                    writer.add_image('sample images ema', ema_grid, global_step=epoch)
                    
                sampled_images = diffusion.sample(model, num_test_images, labels=labels)
                grid = get_grid_images(sampled_images)
                save_images(grid, os.path.join(result_save_dir, f"{epoch}.png"))
                writer.add_image('sample images', grid, global_step=epoch)
            
            save_dict = {
                'model': model.state_dict(),
                'ema_model': ema.state_dict() if use_ema else None,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(save_dict, os.path.join(model_save_dir, f"ckpt.pt"))


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM"
    args.epochs = 500
    args.batch_size = 4
    args.image_size = 64
    args.dataset_path = r"../datasets/Landscape Pictures"
    args.lr = 3e-4
    args.test_term = 10
    args.use_ema = False
    
    train(*args)
    # train(args.run_name,
    #       args.epochs,
    #       args.batch_size,
    #       args.image_size,
    #       args.dataset_path,
    #       args.lr,
    #       args.test_term,
    #       args.use_ema)

if __name__ == "__main__":
    launch()
    