"""Neural network models for audio classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class AudioClassifier(nn.Module):
    """
    CNN-based audio classifier for mel spectrograms.

    Architecture:
    - 4 convolutional blocks with increasing channels
    - Global average pooling
    - Fully connected layers with dropout
    """

    def __init__(
        self,
        num_classes: int,
        n_mels: int = 128,
        channels: list = [32, 64, 128, 256],
        fc_dim: int = 512,
        dropout: float = 0.5,
    ):
        """
        Args:
            num_classes: Number of output classes
            n_mels: Number of mel bands in input
            channels: List of channel sizes for conv blocks
            fc_dim: Dimension of fully connected layer
            dropout: Dropout probability
        """
        super().__init__()

        self.n_mels = n_mels
        self.num_classes = num_classes

        # Convolutional layers
        layers = []
        in_ch = 1  # Single channel input (mono audio)
        for out_ch in channels:
            layers.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input mel spectrogram [batch, 1, n_mels, time] or [batch, n_mels, time]

        Returns:
            Logits [batch, num_classes]
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv_layers(x)
        x = self.gap(x)
        x = self.fc(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class AudioClassifierResNet(nn.Module):
    """
    ResNet-style audio classifier with residual connections.
    Better for deeper networks and longer training.
    """

    def __init__(
        self,
        num_classes: int,
        n_mels: int = 128,
        base_channels: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, num_classes),
        )

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer."""
        layers = []

        # First block may downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """Basic residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)
        return out
