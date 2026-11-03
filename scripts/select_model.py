"""Pick the production model on validation evidence, then calibrate the
human-review threshold on the validation split.

Run after training, before scripts/evaluate.py touches the test split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.evaluation.metrics import directory_size_mb, save_json  # noqa: E402

CANDIDATES = ("SimpleRNN", "LSTM", "GRU", "DistilBERT")


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, bins: int = 10) -> float:
    """ECE over max-probability bins. Reported so the app can say honestly
    whether its confidence numbers are calibrated."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def main() -> int:
    cfg = load_config()
    labels = list(cfg["data"]["labels"])
    metrics_dir = cfg.path("results", "metrics")

    rows = []
    for name in CANDIDATES:
        train_meta = metrics_dir / f"train_{name}.json"
        art = cfg.path("artifacts", name)
        if not train_meta.exists():
            continue
        meta = json.loads(train_meta.read_text())
        rows.append(
            {
                "model": name,
                "val_f1_macro": meta.get("best_val_f1_macro"),
                "train_time_sec": meta.get("train_time_sec"),
                "trainable_parameters": meta.get("trainable_parameters"),
                "artifact_size_mb": directory_size_mb(art) if art.exists() else None,
            }
        )
    if not rows:
        print("no trained models to select from", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (r["val_f1_macro"] or 0.0), reverse=True)
    winner = rows[0]
    print("validation ranking (macro F1):")
    for r in rows:
        print(f"  {r['model']:<11} {r['val_f1_macro']}  {r['artifact_size_mb']} MB")
    print(f"\nselected: {winner['model']}")

    # Threshold calibration on the winner's validation probabilities.
    art = cfg.path("artifacts", winner["model"])
    probs_path, labels_path = art / "val_probs.npy", art / "val_labels.npy"
    threshold_payload: dict
    if probs_path.exists():
        probs = np.load(probs_path)
        if labels_path.exists():
            y_val = np.load(labels_path)
        else:
            import pandas as pd

            val_df = pd.read_parquet(cfg.path(cfg["data"]["processed_dir"], "val.parquet"))
            y_val = val_df["label_id"].to_numpy(dtype=np.int64)
        conf = probs.max(axis=1)
        correct = probs.argmax(axis=1) == y_val

        # Choose the lowest threshold at which the auto-routed slice reaches 90%
        # precision; that is the promise the UI makes when it says "automated".
        target = 0.90
        grid = np.round(np.arange(0.30, 0.991, 0.01), 2)
        sweep = []
        chosen = None
        for t in grid:
            auto = conf >= t
            coverage = float(auto.mean())
            accuracy = float(correct[auto].mean()) if auto.sum() else float("nan")
            sweep.append({"threshold": float(t), "coverage": round(coverage, 4),
                          "auto_accuracy": None if np.isnan(accuracy) else round(accuracy, 4)})
            if chosen is None and auto.sum() >= 50 and accuracy >= target:
                chosen = float(t)
        fallback = float(cfg["inference"]["review_threshold"])
        threshold = chosen if chosen is not None else fallback
        auto = conf >= threshold
        threshold_payload = {
            "model": winner["model"],
            "threshold": threshold,
            "selection_rule": f"lowest threshold with >={target:.0%} accuracy on the auto-routed slice",
            "target_auto_accuracy": target,
            "achieved_auto_accuracy": round(float(correct[auto].mean()), 4) if auto.sum() else None,
            "auto_coverage": round(float(auto.mean()), 4),
            "review_coverage": round(float((~auto).mean()), 4),
            "fell_back_to_config": chosen is None,
            "expected_calibration_error": round(expected_calibration_error(probs, y_val), 4),
            "calibration_note": (
                "These are raw softmax outputs, not calibrated probabilities. The ECE "
                "above is how far the confidence numbers sit from observed accuracy; "
                "the UI reports confidence as a routing signal, not a correctness guarantee."
            ),
            "sweep": sweep,
            "n_val": int(len(y_val)),
        }
        print(
            f"review threshold {threshold:.2f} -> auto coverage "
            f"{threshold_payload['auto_coverage']:.1%}, auto accuracy "
            f"{threshold_payload['achieved_auto_accuracy']}, ECE "
            f"{threshold_payload['expected_calibration_error']}"
        )
    else:
        threshold_payload = {
            "model": winner["model"],
            "threshold": float(cfg["inference"]["review_threshold"]),
            "fell_back_to_config": True,
            "calibration_note": "no validation probabilities saved; using the config default",
        }
        print("WARNING: no val_probs.npy; threshold not calibrated")

    save_json(threshold_payload, metrics_dir / "review_threshold.json")
    save_json(
        {
            "selected_model": winner["model"],
            "decision_basis": "highest validation macro F1, then size and latency",
            "ranking": rows,
            "labels": labels,
        },
        metrics_dir / "model_selection.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
