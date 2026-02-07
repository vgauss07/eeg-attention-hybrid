"""
CBAM: Convolutional Block Attention Module for 1D EEG signals.

Reference: Woo et al. (2018) "CBAM: Convolutional Block Attention Module"
Adapted for 1D: channel attention + temporal attention on (B, C, T) tensors.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention sub-module using avg + max pool."""

    def __init__(self, n_channels: int, reduction: int = 8):
        super().__init__()
        mid = max(n_channels // reduction, 4)
        self.shared_fc = nn.Sequential(
            nn.Linear(n_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → attention weights (B, C, 1)"""
        # Average pool
        avg_pool = x.mean(dim=-1)  # (B, C)
        # Max pool
        max_pool = x.max(dim=-1)[0]  # (B, C)

        avg_out = self.shared_fc(avg_pool)
        max_out = self.shared_fc(max_pool)

        w = torch.sigmoid(avg_out + max_out).unsqueeze(-1)  # (B, C, 1)
        return w


class TemporalAttention(nn.Module):
    """Temporal (spatial in 2D CBAM) attention sub-module for 1D signals."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → temporal attention (B, 1, T)"""
        avg_pool = x.mean(dim=1, keepdim=True)  # (B, 1, T)
        max_pool = x.max(dim=1, keepdim=True)[0]  # (B, 1, T)

        combined = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, T)
        w = torch.sigmoid(self.conv(combined))  # (B, 1, T)
        return w


class CBAM(nn.Module):
    """
    Full CBAM: sequential channel attention → temporal attention.

    Applied to (B, C, T) EEG feature maps.
    """

    def __init__(self, n_channels: int, reduction: int = 8, temporal_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(n_channels, reduction)
        self.temporal_attn = TemporalAttention(temporal_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → refined (B, C, T)"""
        # Channel attention
        ch_w = self.channel_attn(x)  # (B, C, 1)
        x = x * ch_w
        # Temporal attention
        t_w = self.temporal_attn(x)  # (B, 1, T)
        x = x * t_w
        return x

    def get_attention_weights(self, x: torch.Tensor):
        """Return both channel and temporal weights for visualization."""
        ch_w = self.channel_attn(x)  # (B, C, 1)
        x_ch = x * ch_w
        t_w = self.temporal_attn(x_ch)  # (B, 1, T)
        return {
            "channel": ch_w.squeeze(-1),  # (B, C)
            "temporal": t_w.squeeze(1),   # (B, T)
        }
