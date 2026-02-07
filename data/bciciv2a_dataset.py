"""
PyTorch Dataset wrapper for BCI Competition IV 2a preprocessed data.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BCICIV2aDataset(Dataset):
    """
    Loads preprocessed .npz files for a given subject.

    Each .npz contains:
        X_train: (n_train, 22, T) float32
        y_train: (n_train,) int64
        X_test:  (n_test, 22, T) float32
        y_test:  (n_test,) int64
    """

    def __init__(
        self,
        data_dir: str,
        subject_id: int,
        split: str = "train",
        transform=None,
    ):
        self.data_dir = data_dir
        self.subject_id = subject_id
        self.split = split
        self.transform = transform

        fpath = os.path.join(data_dir, f"subject_{subject_id:02d}.npz")
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Preprocessed file not found: {fpath}\n"
                "Run `python -m data.download_bciciv2a` first."
            )

        data = np.load(fpath)
        self.X = torch.from_numpy(data[f"X_{split}"])  # (N, C, T)
        self.y = torch.from_numpy(data[f"y_{split}"])   # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # (C, T)
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_timepoints(self) -> int:
        return self.X.shape[2]

    @property
    def n_classes(self) -> int:
        return len(torch.unique(self.y))


def get_dataloaders(
    data_dir: str,
    subject_id: int,
    batch_size: int = 64,
    num_workers: int = 4,
    transform=None,
) -> Tuple[DataLoader, DataLoader]:
    """Convenience: return train and test DataLoaders for one subject."""
    train_ds = BCICIV2aDataset(data_dir, subject_id, split="train", transform=transform)
    test_ds = BCICIV2aDataset(data_dir, subject_id, split="test", transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


class EEGAugmentation:
    """Simple EEG data augmentations for training."""

    def __init__(
        self,
        noise_std: float = 0.01,
        time_shift_max: int = 50,
        channel_dropout_prob: float = 0.1,
    ):
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.channel_dropout_prob = channel_dropout_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Additive Gaussian noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        # Random time shift
        if self.time_shift_max > 0:
            shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=-1)

        # Random channel dropout
        if self.channel_dropout_prob > 0:
            mask = torch.bernoulli(
                torch.full((x.shape[0], 1), 1 - self.channel_dropout_prob)
            )
            x = x * mask

        return x
