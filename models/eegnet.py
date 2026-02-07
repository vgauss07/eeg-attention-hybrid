"""
EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs.

Reference: Lawhern et al. (2018), J. Neural Engineering.
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al., 2018).

    Architecture:
        1. Temporal conv (1, C, 1, kernel_length) → F1 filters
        2. Depthwise conv (F1, 1, C, 1) → F1*D filters (spatial)
        3. Separable conv (F1*D → F2) with pointwise
        4. Global average pool → FC → softmax
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_timepoints: int = 1000,
        n_classes: int = 4,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # Block 1: Temporal filtering
        self.block1 = nn.Sequential(
            # (B, 1, C, T) → (B, F1, C, T)
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise: (B, F1, C, T) → (B, F1*D, 1, T)
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: Separable conv
        self.block2 = nn.Sequential(
            # Depthwise temporal
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),  # Pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Compute classifier input size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            dummy = self.block1(dummy)
            dummy = self.block2(dummy)
            flat_size = dummy.view(1, -1).shape[1]

        self.classifier = nn.Linear(flat_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw EEG
        Returns:
            logits (B, n_classes)
        """
        # Reshape to (B, 1, C, T) for Conv2d
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return features before classifier for visualization."""
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return x
