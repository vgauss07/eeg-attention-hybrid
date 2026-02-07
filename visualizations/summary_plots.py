"""
Summary plots for ablation study results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_ablation_comparison(
    results: Dict[str, Dict[str, List[float]]],
    metric: str = "accuracy",
    title: str = "Ablation Study — Accuracy by Model",
    save_path: Optional[str] = None,
):
    """
    Box + strip plot comparing models across subjects.

    Args:
        results: Dict mapping model_name → {"accuracy": [per-subject values], ...}
        metric: Which metric to plot.
        title: Plot title.
        save_path: Save path.
    """
    model_names = list(results.keys())
    data = [results[m].get(metric, []) for m in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot
    bp = ax.boxplot(
        data,
        labels=model_names,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
    )

    # Color boxes
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Strip plot (individual subjects)
    for i, (name, vals) in enumerate(zip(model_names, data)):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.full(len(vals), i + 1) + jitter,
            vals,
            alpha=0.6,
            s=30,
            c="black",
            zorder=3,
        )

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_training_curves(
    log_path: str,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
):
    """
    Plot training/test loss and accuracy from a JSON log.

    Args:
        log_path: Path to experiment .json log.
        title: Plot title.
        save_path: Save path.
    """
    with open(log_path) as f:
        log = json.load(f)

    history = log["history"]
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=1.5)
    axes[0].plot(epochs, history["test_loss"], label="Test", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=1.5)
    axes[1].plot(epochs, history["test_acc"], label="Test Acc", linewidth=1.5)
    axes[1].plot(epochs, history["test_balanced_acc"], label="Test Bal. Acc", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    # Learning rate
    axes[2].plot(epochs, history["lr"], color="green", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_confusion_matrices(
    results: Dict[str, np.ndarray],
    class_names: List[str] = None,
    title: str = "Confusion Matrices",
    save_path: Optional[str] = None,
):
    """
    Plot confusion matrices for multiple models side-by-side.

    Args:
        results: Dict mapping model_name → confusion_matrix (n_classes × n_classes)
        class_names: Class labels.
        title: Plot title.
        save_path: Save path.
    """
    if class_names is None:
        class_names = ["Left", "Right", "Feet", "Tongue"]

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, results.items()):
        cm = np.array(cm)
        # Normalize
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

        # Labels
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                        ha="center", va="center", fontsize=8, color=color)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=8, rotation=45)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def generate_results_table(log_dir: str, save_path: Optional[str] = None) -> str:
    """
    Generate a LaTeX-style results table from all experiment logs.

    Returns:
        Formatted string table.
    """
    log_dir = Path(log_dir)
    rows = []

    for fpath in sorted(log_dir.glob("*.json")):
        with open(fpath) as f:
            log = json.load(f)
        fm = log.get("final_metrics", {})
        rows.append({
            "experiment": log.get("experiment_name", fpath.stem),
            "acc": fm.get("accuracy", 0),
            "bal_acc": fm.get("balanced_accuracy", 0),
            "auc": fm.get("roc_auc", 0),
            "params": fm.get("n_params", 0),
            "time": fm.get("training_time_s", 0),
        })

    # Format table
    header = f"{'Experiment':<40} {'Acc':>8} {'Bal.Acc':>8} {'AUC':>8} {'Params':>10} {'Time(s)':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r['experiment']:<40} {r['acc']:>8.4f} {r['bal_acc']:>8.4f} "
            f"{r['auc']:>8.4f} {r['params']:>10,} {r['time']:>8.1f}"
        )

    table = "\n".join(lines)
    print(table)

    if save_path:
        with open(save_path, "w") as f:
            f.write(table)
        print(f"\nSaved table to: {save_path}")

    return table
