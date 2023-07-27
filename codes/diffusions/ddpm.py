import torch
import torch.nn as nn
from tqdm import tqdm
import imageio
import einops
import numpy as np

from diffusions.abstract_diffusion import AbstractDiffusion


class DDPM(AbstractDiffusion):
    def __init__(self, num_time_steps=1000, image_size=256, beta_start=1e-4, beta_end=0.02, device="cuda:0"):
        """
        alpha = 1 - beta_t
        alpha_hat = (1 - beta_1) * (1 - beta_2) * ... * (1 - beta_t)

        self.alpha_hat: [alpha_hat_1, alpha_hat_2, ..., alpha_hat_n] (n: noise_steps)

        Args:
            noise_steps (int, optional): _description_. Defaults to 1000.
            beta_start (_type_, optional): _description_. Defaults to 1e-4.
            beta_end (float, optional): _description_. Defaults to 0.02.
            img_size (int, optional): _description_. Defaults to 256.
            device (str, optional): _description_. Defaults to "cuda".
        """
        super(DDPM, self).__init__(num_time_steps, image_size, beta_start, beta_end, device)
                
        self.betas = self.linear_beta_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def sample(self, model, n, gif_path=None, gif_term = 10):
        """ Sample images from noise images

        z = N(0, I) if t > 1, else 0
        x_t-1 = 1 / sqrt(alphat_t) * (x_t - (1 - alphat_t) / (sqrt(1 - alpha_hat_t)) * model(x_t, t)) + sqrt(beta) * noise

        Args:
            model (torch.nn.Module): image-to-image model. Generally, U-Net is used.
            n (int): batch size

        Returns:
            torch.tensor: sample image
        """
        model.eval()
        frames = []

        with torch.no_grad():
            x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(0, self.num_time_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alphas[t][:, None, None, None]
                alpha_hat = self.alphas_bar[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
                if gif_path is not None:
                    if (i + 1) % gif_term == 0:
                        normalized = x.clone()
                        
                        normalized = (normalized.clamp(-1, 1) + 1) / 2
                        normalized = (normalized * 255).type(torch.uint8)
                        
                        frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n ** 0.5))
                        frame = frame.cpu().numpy().astype(np.uint8)
                        frames.append(frame)
                    
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        
        if gif_path is not None:
            with imageio.get_writer(gif_path, mode='I', fps=len(frames) / 2) as writer:
                for idx, frame in enumerate(frames):
                    writer.append_data(frame)
        
        return x