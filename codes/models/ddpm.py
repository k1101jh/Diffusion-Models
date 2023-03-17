import torch
import torch.nn as nn
from tqdm import tqdm


class DDPM:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, image_size=256, device="cuda"):
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
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """ get tensor of betas

        Returns:
            torch.tensor: [beta_1, beta_2, ..., beta_n] (n: noise_steps)
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        """
        x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * noise

        Args:
            x (torch.tensor): image
            t (int): time step

        Returns:
            torch.tensor: noised image
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample_timesteps(self, batch_size):
        """ Get random timestep
        DDPM trains with x_t (t: random timestep) images.

        Args:
            batch_size (int): batch size

        Returns:
            torch.tensor: random integers between [1, noise_steps] of size (batch_size)
        """
        return torch.randint(low=1, high=self.noise_steps, size=(batch_size,))

    def sample(self, model, n):
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

        with torch.no_grad():
            x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x