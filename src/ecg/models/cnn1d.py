
"""Baseline 1D CNN for five-label PTB-XL ECG classification."""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Convolution, batch normalization, activation and pooling."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        return self.layers(x)


class ECG1DCNN(nn.Module):
    """Small 1D CNN producing five independent diagnostic logits."""

    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.features = nn.Sequential(
            ConvBlock(12, 32, kernel_size=7),
            ConvBlock(32, 64, kernel_size=5),
            ConvBlock(64, 128, kernel_size=3),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """Return logits of shape (batch, num_classes)."""

        if x.ndim != 3 or x.shape[1:] != (12, 1000):
            raise ValueError(
                "Expected input shape (batch, 12, 1000)."
            )

        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
