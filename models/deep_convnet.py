"""
Deep ConvNet for end-to-end EEG decoding.

Reference: Schirrmeister et al. (2017), Human Brain Mapping.
"""

import torch
import torch.nn as nn


class DeepConvNet(nn.Module):
    """
    Deep ConvNet (Schirrmeister et al., 2017).

    Four conv-pool blocks followed by a classification head.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_timepoints: int = 1000,
        n_classes: int = 4,
        n_filters_time: int = 25,
        n_filters_spat: int = 25,
        filter_time_length: int = 10,
        pool_time_length: int = 3,
        pool_time_stride: int = 3,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # Block 1: Temporal → Spatial
        self.block1 = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), bias=False),
            nn.Conv2d(n_filters_time, n_filters_spat, (n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters_spat),
            nn.ELU(),
            nn.MaxPool2d((1, pool_time_length), stride=(1, pool_time_stride)),
            nn.Dropout(dropout_rate),
        )

        # Blocks 2–4: Increasing filters
        n_filters = [n_filters_spat, 50, 100, 200]
        blocks = []
        for i in range(1, len(n_filters)):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        n_filters[i - 1], n_filters[i], (1, filter_time_length), bias=False
                    ),
                    nn.BatchNorm2d(n_filters[i]),
                    nn.ELU(),
                    nn.MaxPool2d((1, pool_time_length), stride=(1, pool_time_stride)),
                    nn.Dropout(dropout_rate),
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # Compute classifier input size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            dummy = self.block1(dummy)
            dummy = self.blocks(dummy)
            flat_size = dummy.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            logits (B, n_classes)
        """
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.block1(x)
        x = self.blocks(x)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return features before classifier."""
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.blocks(x)
        return x
