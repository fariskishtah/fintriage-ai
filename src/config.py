"""Central config loader. Everything reads the same YAML so training,
evaluation and inference cannot drift apart on label order or sequence length."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


class Config(dict):
    """dict with attribute access and resolved project-relative paths."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, *parts: str) -> Path:
        return ROOT.joinpath(*parts)


def load_config(path: str | os.PathLike | None = None) -> Config:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def label_names(cfg: Config | None = None) -> list[str]:
    cfg = cfg or load_config()
    return list(cfg["data"]["labels"])


def label_to_id(cfg: Config | None = None) -> dict[str, int]:
    return {name: i for i, name in enumerate(label_names(cfg))}


def id_to_label(cfg: Config | None = None) -> dict[int, str]:
    return {i: name for i, name in enumerate(label_names(cfg))}


def display_name(label: str, cfg: Config | None = None) -> str:
    cfg = cfg or load_config()
    return cfg["data"]["display_names"].get(label, label)


def set_seed(seed: int) -> None:
    """Seed every RNG we touch. Called at the top of each script."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
