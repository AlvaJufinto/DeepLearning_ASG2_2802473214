import torch
from torch import nn


def build_dcgan_generator(latent_dim=100):
    generator = nn.Sequential(
        # Input: [batch, latent_dim, 1, 1]
        nn.ConvTranspose2d(
            latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False
        ),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        # [batch, 256, 7, 7] -> [batch, 128, 14, 14]
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        # [batch, 128, 14, 14] -> [batch, 1, 28, 28]
        nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
    )

    return generator


def build_dcgan_discriminator():
    discriminator = nn.Sequential(
        # Input: [batch, 1, 28, 28]
        nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # [batch, 128, 14, 14] -> [batch, 256, 7, 7]
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Flatten(),
        nn.Linear(256 * 7 * 7, 1),
    )

    return discriminator
