"""
Train a single model on a single subject.

Usage:
    python scripts/run_single.py --config configs/default.yaml \
        --subject 1 \
        --architecture hybrid_cnn \
        --attention_type se
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from data.bciciv2a_dataset import get_dataloaders
from models.factory import build_model
from trainers.trainer import Trainer
from utils.reproducibility import set_seed, get_device
from visualizations.attention_maps import plot_attention_weights
from visualizations.saliency import compute_integrated_gradients, plot_saliency


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def override_config(cfg: dict, args) -> dict:
    """Apply CLI overrides to config."""
    if args.subject is not None:
        cfg["_subject_id"] = args.subject
    if args.architecture is not None:
        cfg["model"]["architecture"] = args.architecture
    if args.attention_type is not None:
        cfg["model"]["attention_type"] = args.attention_type
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    return cfg


def run(cfg: dict, subject_id: int):
    """Train and evaluate one model on one subject."""
    # Reproducibility
    seed = cfg["reproducibility"]["seed"]
    set_seed(seed, deterministic=cfg["reproducibility"].get("deterministic", True))
    device = get_device()

    # Data
    data_dir = cfg["data"]["processed_dir"]
    train_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        subject_id=subject_id,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["reproducibility"].get("num_workers", 4),
    )

    # Model
    model = build_model(cfg)
    print(f"\nModel: {cfg['model']['architecture']}")
    print(f"Attention: {cfg['model'].get('attention_type', 'none')}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Experiment name
    arch = cfg["model"]["architecture"]
    attn = cfg["model"].get("attention_type", "none")
    exp_name = f"{arch}_{attn}_subject{subject_id:02d}"

    # Train
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        experiment_name=exp_name,
    )
    metrics = trainer.fit()

    # ── Post-training visualizations ──
    fig_dir = cfg["paths"]["figure_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    # Attention weights (if applicable)
    if arch == "hybrid_cnn" and attn != "none":
        model.eval()
        with torch.no_grad():
            sample_x, sample_y = next(iter(test_loader))
            sample_x = sample_x.to(device)
            attn_weights = model.get_attention_weights(sample_x)

        if attn_weights is not None:
            if isinstance(attn_weights, dict):
                # CBAM
                weights_np = {k: v.cpu().numpy() for k, v in attn_weights.items()}
            elif isinstance(attn_weights, torch.Tensor):
                weights_np = attn_weights.cpu().numpy()
            else:
                weights_np = attn_weights

            plot_attention_weights(
                weights_np,
                attention_type=attn,
                title=f"Attention — {exp_name}",
                save_path=os.path.join(fig_dir, f"attn_{exp_name}.png"),
            )

    # Saliency map (for one trial)
    model.eval()
    sample_x, sample_y = next(iter(test_loader))
    trial_x = sample_x[0]  # single trial
    trial_y = sample_y[0].item()

    ig_attrs = compute_integrated_gradients(
        model, trial_x, target_class=trial_y, device=device, n_steps=30,
    )
    plot_saliency(
        ig_attrs,
        title=f"Integrated Gradients — {exp_name} (class={trial_y})",
        save_path=os.path.join(fig_dir, f"saliency_{exp_name}.png"),
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train single EEG model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--attention_type", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = override_config(cfg, args)

    subject_id = cfg.pop("_subject_id", cfg["data"]["subject_ids"][0])

    metrics = run(cfg, subject_id)
    print(f"\n{'='*40}")
    print(f"Final Results for Subject {subject_id}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
