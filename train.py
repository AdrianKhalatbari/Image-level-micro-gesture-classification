"""
Adrian: This is a test just to test training time! You can remove everything and write your own training loop here.

Training script for micro-gesture classification using ResNet18.

This script:
1. Loads the training dataset
2. Builds the ResNet18 model
3. Moves the model to GPU if available
4. Defines loss function and optimizer
5. Trains the model for multiple epochs
6. Saves model checkpoints

Usage:
    python train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_train_loader
from models.resnet18_model import build_resnet18


def train_model(
    train_dir: str,
    num_classes: int = 32,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    save_path: str = "outputs/checkpoints/resnet18_micro_gesture.pth"
):
    """
    Train ResNet18 on the micro-gesture training dataset.

    Parameters
    ----------
    train_dir : str
        Path to the training dataset.

    num_classes : int
        Number of gesture classes.

    batch_size : int
        Number of images per batch.

    learning_rate : float
        Learning rate for Adam optimizer.

    num_epochs : int
        Number of training epochs.

    save_path : str
        Path to save the trained model weights.
    """

    # -----------------------------------------------------------
    # Select device: GPU if available, otherwise CPU
    # -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------------------------------------
    # Load training data
    # -----------------------------------------------------------
    train_loader = get_train_loader(train_dir, batch_size=batch_size)

    # -----------------------------------------------------------
    # Build model and move it to device
    # -----------------------------------------------------------
    model = build_resnet18(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # -----------------------------------------------------------
    # Define loss function and optimizer
    # CrossEntropyLoss is standard for multi-class classification
    # Adam is a good default optimizer
    # -----------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------------------------------------
    # Create output folder if it does not exist
    # -----------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # -----------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------
    for epoch in range(num_epochs):
        model.train()  # set model to training mode

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Move batch to GPU/CPU
            images = images.to(device)
            labels = labels.to(device)

            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_acc:.2f}%"
        )

    # -----------------------------------------------------------
    # Save trained model weights
    # -----------------------------------------------------------
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train_dir = "/content/drive/MyDrive/Colab Notebooks/NVDIA_Project/Image-level-micro-gesture-classification/data/train"

    train_model(
        train_dir=train_dir,
        num_classes=32,
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=10,
        save_path="outputs/checkpoints/resnet18_micro_gesture.pth"
    )