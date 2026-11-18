import torch.nn as nn


def build_vanilla_autoencoder(latent_dim=64):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, latent_dim),
        nn.ReLU(),
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 784),
        nn.Sigmoid(),
    )
