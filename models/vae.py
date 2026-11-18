import torch
from torch import nn


def build_vae(latent_dim=64):
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
    )

    fc_mu = nn.Linear(256, latent_dim)
    fc_log_var = nn.Linear(256, latent_dim)

    decoder = nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 784),
        nn.Sigmoid(),
        nn.Unflatten(1, (1, 28, 28)),
    )

    return encoder, fc_mu, fc_log_var, decoder


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(std)

    return mu + epsilon * std


def forward_vae(images, encoder, fc_mu, fc_log_var, decoder):
    features = encoder(images)

    mu = fc_mu(features)
    log_var = fc_log_var(features)

    latent = reparameterize(mu, log_var)
    reconstructed = decoder(latent)

    return reconstructed, mu, log_var, latent
