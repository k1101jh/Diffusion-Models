from abc import ABCMeta
from abc import abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
import os


class AbstractDiffusion(metaclass=ABCMeta):
    def __init__(self, time_steps, image_size, beta_start, beta_end, device):
        self.num_time_steps = time_steps
        self.image_size = image_size
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
    
    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_time_steps, dtype=torch.float32)
    
    def scaled_linear_beta_schedule(self):
        return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_time_steps, dtype=torch.float32) ** 2
    
    def cosine_beta_schedule(self, s=0.008, max_beta=0.999):
        """
        https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
        https://github.com/vedantroy/improved-ddpm-pytorch/blob/main/diffusion/diffusion.py
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].
        
        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                            produces the cumulative product of (1-beta) up to that
                            part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                            prevent singularities.
        """
        steps = self.num_time_steps + 1
        x = torch.linspace(0, self.num_time_steps, steps)
        alphas_cumprod = torch.cos(((x / self.num_time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(max=max_beta)
    
    def sample_timesteps(self, batch_size):
        """ Get random timestep
        DDPM trains with x_t (t: random timestep) images.

        Args:
            batch_size (int): batch size

        Returns:
            torch.tensor: random integers between [1, noise_steps] of size (batch_size)
        """
        return torch.randint(low=1, high=self.num_time_steps, size=(batch_size,))
    
    def noise_images(self, x, t):
        """
        x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * noise

        Args:
            x (torch.tensor): image
            t (int): time step

        Returns:
            torch.tensor: noised image
        """
        sqrt_alpha_hat = torch.sqrt(self.alphas_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alphas_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e
    
    @abstractmethod
    def sample(self, model, n):
        raise NotImplementedError()
    
    @classmethod
    def from_config(cls, config):
        assert(cls is not None)
        
        return cls(**config)
    
    def save_config(self, save_path):
        assert(os.path.isfile(save_path))
        
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        