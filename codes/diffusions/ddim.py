import torch
import torch.nn as nn
from tqdm import tqdm
import imageio
import einops
import numpy as np

from diffusions.abstract_diffusion import AbstractDiffusion


class DDIM(AbstractDiffusion):
    def __init__(self, num_time_steps=1000, image_size=256, beta_start=1e-4, beta_end=0.02, device="cuda:0"):
        super(DDIM, self).__init__(num_time_steps, image_size, beta_start, beta_end, device)
        
        # alpha = 1 - beta
        # alpha_bar = alpha_1 * alpha_2 * ... * alpha_t
        self.betas = self.linear_beta_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0, device=device)
        
    def _get_std(self, alpha_bar, alpha_bar_prev):
        """
        σ_t = sqrt((1 - α_t-1)/(1 - α_t)) * sqrt(1 - α_t/α_t-1)
        """
        return torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
        
    
    def sample(self, model, n, num_sampling_steps, eta=0., gif_path=None, gif_term=1, clip_sample=False):
        model.eval()
        frames = []

        with torch.no_grad():
            x = torch.randn((n, 3, self.image_size, self.image_size), device=self.device)
            sampling_steps = list(reversed(range(0, self.num_time_steps, self.num_time_steps // num_sampling_steps)))
            for idx, t in tqdm(enumerate(sampling_steps), position=0):
                predicted_noise = model(x, (torch.ones(n) * t).long().to(self.device))
                alpha_bar = self.alphas_bar[t]
                
                # prev_t가 0보다 작아지는 경우 alpha_bar_prev = 1로 설정
                if idx + 1 < len(sampling_steps):
                    prev_t = sampling_steps[idx + 1]
                    alpha_bar_prev = self.alphas_bar[prev_t]
                else:
                    alpha_bar_prev = self.final_alpha_cumprod
                
                std = eta * self._get_std(alpha_bar, alpha_bar_prev)
                predicted_x0 = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
                direction_pointing_to_xt = torch.sqrt(1 - alpha_bar_prev - std ** 2) * predicted_noise
                
                if clip_sample:
                    predicted_x0 = torch.clamp(predicted_x0, -1, 1)
                
                x = torch.sqrt(alpha_bar_prev) * predicted_x0 + direction_pointing_to_xt
                
                # Add noise
                if eta > 0:
                    if t > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x += std * noise
                
                if gif_path is not None:
                    if (t + 1) % gif_term == 0:
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