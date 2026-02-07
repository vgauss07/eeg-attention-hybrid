"""
Saliency-based interpretability for EEG models.

Implements:
- Integrated Gradients (Sundararajan et al., 2017)
- Simple gradient saliency
- Grad-CAM adapted for 1D temporal signals
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def compute_integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor = None,
    n_steps: int = 50,
    device: torch.device = None,
) -> np.ndarray:
    """
    Compute Integrated Gradients for a single EEG trial.

    Args:
        model: Trained model.
        x: Single input tensor (C, T).
        target_class: Target class index.
        baseline: Baseline input (default: zeros).
        n_steps: Number of interpolation steps.
        device: Computation device.

    Returns:
        Attributions array (C, T).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    x = x.unsqueeze(0).to(device)  # (1, C, T)
    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.unsqueeze(0).to(device)

    # Interpolation
    alphas = torch.linspace(0, 1, n_steps + 1, device=device)
    grads = []

    for alpha in alphas:
        interp = baseline + alpha * (x - baseline)
        interp.requires_grad_(True)

        logits = model(interp)
        score = logits[0, target_class]

        model.zero_grad()
        score.backward()

        grads.append(interp.grad.detach().cpu())

    # Average gradients × (input - baseline)
    grads = torch.stack(grads)  # (n_steps+1, 1, C, T)
    avg_grads = grads.mean(dim=0)
    attributions = (x.cpu() - baseline.cpu()) * avg_grads

    return attributions.squeeze(0).numpy()  # (C, T)


def compute_simple_gradient(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    device: torch.device = None,
) -> np.ndarray:
    """Simple input × gradient saliency."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = x.unsqueeze(0).to(device).requires_grad_(True)

    logits = model(x)
    score = logits[0, target_class]

    model.zero_grad()
    score.backward()

    saliency = (x * x.grad).detach().cpu().squeeze(0).numpy()
    return saliency


def plot_saliency(
    saliency: np.ndarray,
    title: str = "Saliency Map",
    channel_names: list = None,
    sfreq: int = 250,
    tmin: float = 0.5,
    save_path: Optional[str] = None,
):
    """
    Plot saliency heatmap (channels × time).

    Args:
        saliency: (C, T) attribution array.
        title: Plot title.
        channel_names: List of channel names.
        sfreq: Sampling frequency (for time axis).
        tmin: Start time in seconds.
        save_path: Where to save.
    """
    C, T = saliency.shape
    times = np.linspace(tmin, tmin + T / sfreq, T)

    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(C)]

    # Normalize
    vmax = np.percentile(np.abs(saliency), 99)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        saliency,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[times[0], times[-1], C - 0.5, -0.5],
        interpolation="bilinear",
    )

    ax.set_yticks(range(C))
    ax.set_yticklabels(channel_names[:C], fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Attribution", fraction=0.02, pad=0.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
