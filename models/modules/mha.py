"""
Lightweight Multi-Head Attention (MHA) for EEG feature reweighting.

Applies self-attention across the channel dimension of (B, C, T) tensors,
treating each channel as a token with T-dimensional features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightMHA(nn.Module):
    """
    Multi-head self-attention across EEG channels.

    Input:  (B, C, T) — C channels, T time features
    Output: (B, C, T) — attention-refined features

    Each channel is treated as a token. Linear projections reduce
    the time dimension to d_model before computing attention, then
    project back.
    """

    def __init__(
        self,
        n_channels: int,
        t_dim: int,
        n_heads: int = 4,
        d_model: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Project from T → d_model
        self.proj_in = nn.Linear(t_dim, d_model)

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection back to T
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, t_dim),
            nn.Dropout(dropout),
        )

        # Layer norm + residual
        self.norm = nn.LayerNorm(t_dim)
        self.dropout = nn.Dropout(dropout)

        # Store attention weights for visualization
        self._attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T) with residual connection
        """
        B, C, T = x.shape
        residual = x

        # Project: (B, C, T) → (B, C, d_model)
        h = self.proj_in(x)

        # Multi-head Q, K, V
        Q = self.W_q(h).view(B, C, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, C, d_k)
        K = self.W_k(h).view(B, C, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(h).view(B, C, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, C, C)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Store for visualization
        self._attention_weights = attn.detach()

        # Attend
        out = torch.matmul(attn, V)  # (B, H, C, d_k)
        out = out.transpose(1, 2).contiguous().view(B, C, self.d_model)  # (B, C, d_model)

        # Project back to T
        out = self.proj_out(out)  # (B, C, T)

        # Residual + norm
        out = self.norm(out + residual)
        return out

    def get_attention_weights(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Return attention weights (B, H, C, C).
        If x is provided, run a forward pass first.
        """
        if x is not None:
            with torch.no_grad():
                self.forward(x)
        return self._attention_weights
