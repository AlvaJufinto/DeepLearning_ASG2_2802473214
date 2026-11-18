import torch.nn as nn


def build_mlp_classifier():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def extract_features(model, images):
    return model[:5](images)
