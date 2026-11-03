"""Shared training helpers: device choice, class weights, loaders, early stopping."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    path = Path(processed_dir) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/prepare_data.py first")
    return pd.read_parquet(path)


def balanced_class_weights(label_ids: np.ndarray, num_classes: int) -> np.ndarray:
    """sklearn's 'balanced' formula: n / (k * count_c). Training split only."""
    counts = np.bincount(label_ids, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    return len(label_ids) / (num_classes * counts)


@dataclass
class EarlyStopping:
    patience: int = 2
    mode: str = "max"
    best: float = field(default=float("-inf"))
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.mode == "min":
            self.best = float("inf")

    def step(self, value: float) -> tuple[bool, bool]:
        """Return (improved, should_stop)."""
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience
