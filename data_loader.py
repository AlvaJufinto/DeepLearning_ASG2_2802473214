import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split

from utils.paths import DATA_DIR


def get_dataloaders(root=DATA_DIR, batch_size=128, seed=42):
    transform = transforms.ToTensor()

    train_data = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )

    test_data = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )

    # Combine official train and test datasets
    full_dataset = ConcatDataset([train_data, test_data])

    # 80:10:10 split
    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))

    print("Dataset successfully loaded!")
    print(f"Train: {len(train_loader.dataset)}")
    print(f"Validation: {len(val_loader.dataset)}")
    print(f"Test: {len(test_loader.dataset)}")
    print(f"Image shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Pixel range: {images.min():.2f} - {images.max():.2f}")
