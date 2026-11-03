"""Metric computation shared by every model, so the comparison table is
produced by one code path rather than four."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict:
    ids = list(range(len(labels)))
    per_class = classification_report(
        y_true, y_pred, labels=ids, target_names=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(per_class[name]["precision"]),
                "recall": float(per_class[name]["recall"]),
                "f1": float(per_class[name]["f1-score"]),
                "support": int(per_class[name]["support"]),
            }
            for name in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
        "n_samples": int(len(y_true)),
    }


def measure_cpu_latency(predict_fn, samples: list[str], warmup: int = 3, repeats: int = 30) -> dict:
    """Single-example CPU latency, which is what the deployed app actually does."""
    for i in range(min(warmup, len(samples))):
        predict_fn(samples[i % len(samples)])
    timings = []
    for i in range(repeats):
        start = time.perf_counter()
        predict_fn(samples[i % len(samples)])
        timings.append((time.perf_counter() - start) * 1000.0)
    arr = np.array(timings)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "repeats": repeats,
    }


def directory_size_mb(path: str | Path) -> float:
    path = Path(path)
    if path.is_file():
        return round(path.stat().st_size / 1e6, 2)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1e6, 2)


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
