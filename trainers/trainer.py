"""
Trainer: handles training loop, evaluation, logging, checkpointing.
"""

import os
import time
import json
import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

from utils.metrics import compute_metrics
from utils.reproducibility import set_seed


class Trainer:
    """
    Training and evaluation manager.

    Handles:
        - Training loop with gradient clipping
        - Validation / test evaluation
        - Early stopping
        - Checkpoint saving/loading
        - Per-epoch metric logging
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device = None,
        experiment_name: str = "experiment",
    ):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = experiment_name

        self.model.to(self.device)

        # Training config
        tcfg = cfg["training"]
        self.epochs = tcfg["epochs"]
        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.early_stopping_patience = tcfg.get("early_stopping_patience", 50)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self._build_optimizer(tcfg)

        # Scheduler
        self.scheduler = self._build_scheduler(tcfg)

        # Paths
        pcfg = cfg["paths"]
        self.checkpoint_dir = Path(pcfg["checkpoint_dir"])
        self.log_dir = Path(pcfg["log_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "test_balanced_acc": [],
            "test_roc_auc": [],
            "lr": [],
        }
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_state = None
        self.patience_counter = 0

    def _build_optimizer(self, tcfg: dict) -> torch.optim.Optimizer:
        opt_name = tcfg.get("optimizer", "adamw").lower()
        lr = tcfg["lr"]
        wd = tcfg.get("weight_decay", 1e-4)

        if opt_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self, tcfg: dict):
        sched_name = tcfg.get("scheduler", "cosine").lower()
        if sched_name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=tcfg["epochs"], eta_min=1e-6)
        elif sched_name == "step":
            return StepLR(self.optimizer, step_size=50, gamma=0.5)
        elif sched_name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5,
                patience=tcfg.get("scheduler_patience", 20),
            )
        else:
            return None

    def train_epoch(self) -> Tuple[float, float]:
        """Run one training epoch. Returns (loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader: DataLoader = None) -> Dict[str, float]:
        """Evaluate on test set. Returns dict of metrics."""
        if loader is None:
            loader = self.test_loader

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        total = 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            loss = self.criterion(logits, y)

            total_loss += loss.item() * X.size(0)
            total += X.size(0)

            probs = torch.softmax(logits, dim=1)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        metrics = compute_metrics(all_labels, all_preds, all_probs)
        metrics["loss"] = total_loss / total
        return metrics

    def fit(self) -> Dict[str, float]:
        """
        Full training loop with early stopping.

        Returns:
            Best test metrics dict.
        """
        print(f"\n{'='*60}")
        print(f"Training: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch()

            # Evaluate
            test_metrics = self.evaluate()

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(test_metrics["accuracy"])
                else:
                    self.scheduler.step()

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_metrics["loss"])
            self.history["test_acc"].append(test_metrics["accuracy"])
            self.history["test_balanced_acc"].append(test_metrics["balanced_accuracy"])
            self.history["test_roc_auc"].append(test_metrics.get("roc_auc", 0.0))
            self.history["lr"].append(current_lr)

            # Print
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:>4d}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Test Acc: {test_metrics['accuracy']:.4f} "
                    f"Bal: {test_metrics['balanced_accuracy']:.4f} | "
                    f"LR: {current_lr:.2e}"
                )

            # Best model tracking
            if test_metrics["accuracy"] > self.best_acc:
                self.best_acc = test_metrics["accuracy"]
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⏹  Early stopping at epoch {epoch} (patience={self.early_stopping_patience})")
                break

        elapsed = time.time() - start_time

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        # Final evaluation with best model
        final_metrics = self.evaluate()
        final_metrics["best_epoch"] = self.best_epoch
        final_metrics["training_time_s"] = elapsed
        final_metrics["n_params"] = sum(p.numel() for p in self.model.parameters())

        print(f"\n✓ Training complete in {elapsed:.1f}s")
        print(f"  Best epoch: {self.best_epoch}")
        print(f"  Test Accuracy:          {final_metrics['accuracy']:.4f}")
        print(f"  Test Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
        print(f"  Test ROC-AUC:           {final_metrics.get('roc_auc', 0.0):.4f}")

        # Save
        self._save_checkpoint(final_metrics)
        self._save_log(final_metrics)

        return final_metrics

    def _save_checkpoint(self, metrics: dict):
        path = self.checkpoint_dir / f"{self.experiment_name}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.cfg,
            },
            path,
        )

    def _save_log(self, final_metrics: dict):
        log = {
            "experiment_name": self.experiment_name,
            "final_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                              for k, v in final_metrics.items()},
            "history": {k: [float(v) for v in vals] for k, vals in self.history.items()},
            "config": self.cfg,
        }
        path = self.log_dir / f"{self.experiment_name}.json"
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
