"""
Hybrid CNN + Attention Model for EEG Decoding.

Pluggable attention: none, SE, CBAM, or lightweight MHA.
This is the core proposed architecture from the paper.
"""

import torch
import torch.nn as nn

from .modules import SEBlock, CBAM, LightweightMHA


class HybridCNN(nn.Module):
    """
    Hybrid CNN + Attention for EEG classification.

    Pipeline:
        1. Temporal convolutions (1D across time)
        2. Spatial / depthwise convolutions (channel mixing)
        3. Attention module (SE / CBAM / MHA / none)
        4. Global average pooling → FC → softmax

    Args:
        n_channels:        Number of EEG channels (e.g. 22)
        n_timepoints:      Number of time samples per trial
        n_classes:         Number of output classes
        attention_type:    "none", "se", "cbam", or "mha"
        temporal_filters:  Number of temporal conv filters
        temporal_kernel:   Kernel size for temporal conv
        spatial_filters:   Number of spatial conv filters
        pool_size:         Pooling kernel size
        dropout:           Dropout rate
        se_reduction:      Reduction ratio for SE block
        mha_n_heads:       Number of heads for MHA
        mha_d_model:       Model dimension for MHA
    """

    VALID_ATTENTION = {"none", "se", "cbam", "mha"}

    def __init__(
        self,
        n_channels: int = 22,
        n_timepoints: int = 1000,
        n_classes: int = 4,
        attention_type: str = "se",
        temporal_filters: int = 16,
        temporal_kernel: int = 64,
        spatial_filters: int = 32,
        pool_size: int = 8,
        dropout: float = 0.5,
        se_reduction: int = 8,
        mha_n_heads: int = 4,
        mha_d_model: int = 32,
    ):
        super().__init__()
        assert attention_type in self.VALID_ATTENTION, (
            f"attention_type must be one of {self.VALID_ATTENTION}, got '{attention_type}'"
        )
        self.attention_type = attention_type
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # ── Stage 1: Temporal convolutions ──
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, temporal_filters, kernel_size=temporal_kernel,
                      padding=temporal_kernel // 2, bias=False),
            nn.BatchNorm1d(temporal_filters),
            nn.ELU(),
            nn.AvgPool1d(pool_size),
            nn.Dropout(dropout),
        )

        t_after_pool = n_timepoints // pool_size

        # ── Stage 2: Spatial / depthwise convolutions ──
        self.spatial = nn.Sequential(
            # Depthwise-style: each temporal filter independently
            nn.Conv1d(temporal_filters, spatial_filters, kernel_size=16,
                      padding=8, groups=1, bias=False),
            nn.BatchNorm1d(spatial_filters),
            nn.ELU(),
            nn.AvgPool1d(pool_size),
            nn.Dropout(dropout),
        )

        t_after_spatial = t_after_pool // pool_size

        # ── Stage 3: Attention module ──
        self.attention = self._build_attention(
            attention_type=attention_type,
            n_features=spatial_filters,
            t_dim=t_after_spatial,
            se_reduction=se_reduction,
            mha_n_heads=mha_n_heads,
            mha_d_model=mha_d_model,
            dropout=dropout,
        )

        # ── Stage 4: Classification head ──
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, spatial_filters, 1)
            nn.Flatten(),
            nn.Linear(spatial_filters, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    @staticmethod
    def _build_attention(
        attention_type: str,
        n_features: int,
        t_dim: int,
        se_reduction: int,
        mha_n_heads: int,
        mha_d_model: int,
        dropout: float,
    ) -> nn.Module:
        if attention_type == "none":
            return nn.Identity()
        elif attention_type == "se":
            return SEBlock(n_channels=n_features, reduction=se_reduction)
        elif attention_type == "cbam":
            return CBAM(n_channels=n_features, reduction=se_reduction)
        elif attention_type == "mha":
            return LightweightMHA(
                n_channels=n_features,
                t_dim=t_dim,
                n_heads=mha_n_heads,
                d_model=mha_d_model,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — raw/filtered EEG
        Returns:
            logits (B, n_classes)
        """
        x = self.temporal(x)   # (B, temporal_filters, T')
        x = self.spatial(x)    # (B, spatial_filters, T'')
        x = self.attention(x)  # (B, spatial_filters, T'')
        x = self.head(x)       # (B, n_classes)
        return x

    def get_features_before_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Return features right before the attention module."""
        x = self.temporal(x)
        x = self.spatial(x)
        return x

    def get_attention_weights(self, x: torch.Tensor):
        """Extract attention weights for interpretability."""
        features = self.get_features_before_attention(x)
        if self.attention_type == "none":
            return None
        elif hasattr(self.attention, "get_attention_weights"):
            return self.attention.get_attention_weights(features)
        return None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
