import torch
from torch import nn


def build_generator(latent_dim=100):
    generator = nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(1024, 784),
        nn.Tanh(),
        nn.Unflatten(1, (1, 28, 28)),
    )

    return generator


def build_discriminator():
    discriminator = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 1),
    )

    return discriminator
