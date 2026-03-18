"""
Dataset and DataLoader utilities for the micro-gesture classification.

This module:
1. Defines preprocessing and augmentation transforms for training images.
2. Loads the dataset using torchvision.datasets.ImageFolder.
3. Creates a PyTorch DataLoader for training.

Expected dataset structure:

data/
 └── train/
      ├── 1/
      │     ├── img1.jpg
      │     ├── img2.jpg
      │     └── ...
      ├── 2/
      │     └── ...
      └── ...

Each folder name represents a class label.
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# ======================== Image preprocessing and augmentation ========================
# These transforms are applied to every image before it enters the neural network.
#
# Resize: ensures all images have the same size (224x224)
# RandomHorizontalFlip: data augmentation to improve generalization
# RandomRotation: small rotations to make the model more robust
# ToTensor: converts image to PyTorch tensor
# Normalize: standard ImageNet normalization (required for ResNet)
# -----------------------------------------------------------

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])


# ======================== Dataset loader ========================
def get_train_dataset(train_dir: str):
    """
    Loads the training dataset using ImageFolder.

    Parameters
    ----------
    train_dir : str
        Path to the training dataset directory.

    Returns
    -------
    dataset : torchvision.datasets.ImageFolder
        Dataset object containing images and labels.
    """

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transforms
    )

    return dataset


# ======================= DataLoader ========================
def get_train_loader(train_dir: str, batch_size: int = 32, val_split: float = 0):
    """
    Creates a DataLoader for the training dataset.

    Parameters
    ----------
    train_dir : str
        Path to the training dataset directory.

    batch_size : int, optional
        Number of samples per batch (default = 32).

    Returns
    -------
    loader : torch.utils.data.DataLoader
        DataLoader that yields batches of images and labels.
    """

    dataset = get_train_dataset(train_dir)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader