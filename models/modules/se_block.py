"""
Squeeze-and-Excitation (SE) Block for EEG feature reweighting.

Reference: Hu et al. (2018) "Squeeze-and-Excitation Networks"
Adapted for 1D EEG signals: operates on (B, C, T) tensors.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.

    Learns channel-wise attention weights by:
    1. Squeeze: global average pooling over time → (B, C, 1)
    2. Excitation: FC → ReLU → FC → Sigmoid → (B, C, 1)
    3. Scale: element-wise multiply with input
    """

    def __init__(self, n_channels: int, reduction: int = 8):
        super().__init__()
        mid = max(n_channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(n_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            Reweighted tensor (B, C, T)
        """
        b, c, _ = x.size()
        # Squeeze
        w = self.squeeze(x).view(b, c)  # (B, C)
        # Excitation
        w = self.excitation(w).view(b, c, 1)  # (B, C, 1)
        # Scale
        return x * w

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights for visualization."""
        b, c, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w)  # (B, C)
        return w
