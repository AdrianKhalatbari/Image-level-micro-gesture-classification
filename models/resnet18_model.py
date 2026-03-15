"""
Defines the CNN model used for micro-gesture classification.

We use ResNet18 with transfer learning:
• Load a pretrained ResNet18 (trained on ImageNet)
• Replace the final fully connected layer
• Adapt it to our dataset with 32 gesture classes

ResNet18 is chosen because:
- lightweight and fast
- good performance for image classification
- widely used baseline model
"""

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 32, pretrained: bool = True):
    """
    Build a ResNet18 model for micro-gesture classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes in the dataset (default = 32).

    pretrained : bool
        If True, load ImageNet pretrained weights.
        Transfer learning improves performance and speeds up training.

    Returns
    -------
    model : torch.nn.Module
        ResNet18 model with modified classification layer.
    """


    # ======================== Load the pretrained ResNet18 architecture ========================
    model = models.resnet18(pretrained=pretrained)

    # ======================== Get number of input features of the final layer ========================
    in_features = model.fc.in_features

    # ======================== Replace the final fully connected layer ========================
    #
    # Original:
    #   512 -> 1000 classes (ImageNet)
    #
    # New:
    #   512 -> 32 classes (micro-gesture dataset)

    model.fc = nn.Linear(in_features, num_classes)

    return model