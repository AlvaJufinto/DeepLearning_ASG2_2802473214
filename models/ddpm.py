import torch
from torch import nn


def build_noise_predictor():
    model = nn.Sequential(
        nn.Conv2d(2, 64, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, 128),
        nn.SiLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.GroupNorm(8, 128),
        nn.SiLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, 64),
        nn.SiLU(),
        nn.Conv2d(64, 1, kernel_size=3, padding=1),
    )

    return model


def predict_noise(noisy_images, timesteps, model, total_timesteps):
    batch_size = noisy_images.shape[0]

    timestep_channel = (timesteps.float() / (total_timesteps - 1)).view(
        batch_size, 1, 1, 1
    )

    timestep_channel = timestep_channel.expand(
        -1, 1, noisy_images.shape[2], noisy_images.shape[3]
    )

    model_input = torch.cat([noisy_images, timestep_channel], dim=1)

    return model(model_input)
