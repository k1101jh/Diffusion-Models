import os
from accelerate import Accelerator
import torch
from ema_pytorch import EMA
import random
import numpy as np

from backbones.unet import UNet
from diffusions.ddpm import DDPM
from diffusions.ddim import DDIM
from utils import get_grid_images, plot_images, save_images

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


if __name__ == "__main__":
    ## Edit space ##
    load_run_name = "DDPM"
    diffusion_constructor = DDIM
    model_name = "ckpt.pt"
    use_ema_model = False
    num_time_steps = 1000
    num_sampling_steps = 100
    seed = 0
    ################
    
    # Set Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load Diffusion model
    saved_model_dir = os.path.join("./saved_models", load_run_name)
    saved_model_path = os.path.join(saved_model_dir, model_name)
    
    # Set sample save path
    if use_ema_model:
        sample_save_dir = os.path.join("./samples", diffusion_constructor.__name__, "ema_model")
    else:
        sample_save_dir = os.path.join("./samples", diffusion_constructor.__name__, "model")
    
    os.makedirs(sample_save_dir, exist_ok=True)
    
    sample_file_list = os.listdir(sample_save_dir)
    sample_idx = 1
    sample_file_name = f"sampling_steps_{num_sampling_steps}_{sample_idx}.png"
    while sample_file_name in sample_file_list:
        sample_idx += 1
        sample_file_name = f"sampling_steps_{num_sampling_steps}_{sample_idx}.png"
    sample_gif_name = f"sampling_steps_{num_sampling_steps}_{sample_idx}.gif"
    
    sample_save_path = os.path.join(sample_save_dir, sample_file_name)
    sample_gif_path = os.path.join(sample_save_dir, sample_gif_name)
    
    # Sampling
    accelerator = Accelerator()
    device = accelerator.device
    model = UNet().to(device)
    ckpt = torch.load(saved_model_path)
    if use_ema_model:
        ema = EMA(
            model,
        )
        ema.to(device)
        ema.load_state_dict(ckpt["ema_model"])
        model = ema.ema_model
    else:
        model.load_state_dict(ckpt["model"])
    diffusion = diffusion_constructor(num_time_steps=num_time_steps, image_size=64, device=device)
    x = diffusion.sample(model, n=16, num_sampling_steps=num_sampling_steps, gif_path=sample_gif_path, clip_sample=True)
    
    # Save samples
    grid = get_grid_images(x)
    # plot_images(grid)
    save_images(grid, sample_save_path)
    print(f"Sample saved! path: {sample_save_path}\tgif path: {sample_gif_path}")