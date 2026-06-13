import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Remplace la dernière couche fully-connected pour notre nombre de classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model