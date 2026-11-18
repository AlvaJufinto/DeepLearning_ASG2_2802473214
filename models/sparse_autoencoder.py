import torch
from torch import nn


def build_sparse_autoencoder(latent_dim=64):
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, latent_dim),
        nn.ReLU(),
    )

    decoder = nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 784),
        nn.Sigmoid(),
        nn.Unflatten(1, (1, 28, 28)),
    )

    return encoder, decoder
