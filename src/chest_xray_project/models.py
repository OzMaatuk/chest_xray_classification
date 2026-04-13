from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from chest_xray_project.config import ModelConfig


class ChestXrayCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def build_efficientnet(num_classes: int, freeze_backbone: bool, finetune_blocks: list[int]):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    for block_index in finetune_blocks:
        for param in model.features[block_index].parameters():
            param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )
    return model


class PatchLSTMClassifier(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels

        self.projection = nn.Linear(patch_dim, embed_dim)
        self.batch_norm = nn.BatchNorm1d(self.num_patches)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        batch_size, channels, _, _ = inputs.shape
        patch = self.patch_size
        tokens = inputs.unfold(2, patch, patch).unfold(3, patch, patch)
        tokens = tokens.permute(0, 2, 3, 1, 4, 5).contiguous()
        tokens = tokens.view(batch_size, -1, channels * patch * patch)
        tokens = self.projection(tokens)
        tokens = self.batch_norm(tokens)
        outputs, _ = self.lstm(tokens)
        return self.classifier(outputs.mean(dim=1))


def build_model(config: ModelConfig, image_size: int) -> nn.Module:
    if config.name == "cnn":
        return ChestXrayCNN(num_classes=config.num_classes)
    if config.name == "efficientnet":
        return build_efficientnet(
            num_classes=config.num_classes,
            freeze_backbone=config.freeze_backbone,
            finetune_blocks=config.finetune_blocks,
        )
    if config.name == "patch_lstm":
        return PatchLSTMClassifier(
            img_size=image_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model name: {config.name}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

