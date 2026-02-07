"""
Attention weight visualization for EEG models.

Includes:
- Channel attention bar plots
- Temporal attention heatmaps
- MHA attention matrices
- Topographic maps of channel attention (requires MNE)
"""

from typing import Optional, Dict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# Standard 10-20 channel names for BCI-IV-2a (22 channels)
BCICIV2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]


def plot_attention_weights(
    weights,
    attention_type: str,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
    channel_names: list = None,
):
    """
    Plot attention weights based on attention type.

    Args:
        weights: Attention weights (varies by type)
        attention_type: "se", "cbam", or "mha"
        title: Plot title
        save_path: If provided, save figure here
        channel_names: Channel labels
    """
    if channel_names is None:
        channel_names = BCICIV2A_CHANNELS

    if attention_type == "se":
        _plot_se_weights(weights, title, save_path, channel_names)
    elif attention_type == "cbam":
        _plot_cbam_weights(weights, title, save_path, channel_names)
    elif attention_type == "mha":
        _plot_mha_weights(weights, title, save_path, channel_names)


def _plot_se_weights(
    weights: np.ndarray,
    title: str,
    save_path: Optional[str],
    channel_names: list,
):
    """
    SE attention weights: bar chart of channel importance.

    weights: (n_features,) — averaged across batch
    """
    if weights.ndim > 1:
        weights = weights.mean(axis=0)

    n = len(weights)
    labels = channel_names[:n] if len(channel_names) >= n else [f"F{i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.RdYlBu_r(weights / weights.max())
    ax.bar(range(n), weights, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel("Attention Weight")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def _plot_cbam_weights(
    weights: Dict[str, np.ndarray],
    title: str,
    save_path: Optional[str],
    channel_names: list,
):
    """
    CBAM weights: channel bar + temporal heatmap.

    weights: dict with "channel" (n_features,) and "temporal" (T,)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 1]})

    # Channel attention
    ch_w = weights["channel"]
    if ch_w.ndim > 1:
        ch_w = ch_w.mean(axis=0)
    n = len(ch_w)
    labels = channel_names[:n] if len(channel_names) >= n else [f"F{i}" for i in range(n)]

    axes[0].bar(range(n), ch_w, color="steelblue", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(labels, rotation=45, fontsize=8)
    axes[0].set_ylabel("Channel Attention")
    axes[0].set_title(f"{title} — Channel Attention")

    # Temporal attention
    t_w = weights["temporal"]
    if t_w.ndim > 1:
        t_w = t_w.mean(axis=0)
    axes[1].plot(t_w, color="darkorange", linewidth=1.5)
    axes[1].fill_between(range(len(t_w)), t_w, alpha=0.3, color="darkorange")
    axes[1].set_xlabel("Time Step (post-pooling)")
    axes[1].set_ylabel("Temporal Attention")
    axes[1].set_title(f"{title} — Temporal Attention")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def _plot_mha_weights(
    weights: np.ndarray,
    title: str,
    save_path: Optional[str],
    channel_names: list,
):
    """
    MHA attention matrices: one heatmap per head.

    weights: (n_heads, C, C) — averaged across batch
    """
    if weights.ndim == 4:
        weights = weights.mean(axis=0)  # average over batch

    n_heads = weights.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for h in range(n_heads):
        im = axes[h].imshow(weights[h], cmap="viridis", aspect="auto")
        axes[h].set_title(f"Head {h + 1}")
        axes[h].set_xlabel("Key Channel")
        axes[h].set_ylabel("Query Channel")
        plt.colorbar(im, ax=axes[h], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_channel_attention_topomap(
    weights: np.ndarray,
    title: str = "Channel Attention Topomap",
    save_path: Optional[str] = None,
):
    """
    Plot channel attention as a topographic map (requires MNE).

    weights: (22,) attention values for each BCI-IV-2a channel
    """
    try:
        import mne
    except ImportError:
        print("MNE not available — skipping topomap.")
        return

    # Create MNE info with standard 10-20 montage
    info = mne.create_info(
        ch_names=BCICIV2A_CHANNELS,
        sfreq=250,
        ch_types="eeg",
    )
    montage = mne.channels.make_standard_montage("standard_1020")

    # Some channels might not match standard 1020 exactly
    info.set_montage(montage, on_missing="warn")

    fig, ax = plt.subplots(figsize=(6, 5))
    mne.viz.plot_topomap(
        weights[:22],
        info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
    )
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
