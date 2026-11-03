"""Final evaluation of every trained model on the same held-out test split.

    python scripts/evaluate.py

Writes results/metrics/test_<model>.json, results/metrics/model_comparison.csv
and results/metrics/model_comparison.json. Nothing here touches the test split
until the models are already trained and selected on validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, set_seed  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_metrics,
    directory_size_mb,
    measure_cpu_latency,
    save_json,
)
from src.models.rnn import RecurrentClassifier  # noqa: E402
from src.preprocessing.vectorizer import WordVectorizer  # noqa: E402
from src.training.common import load_split  # noqa: E402

RNN_CELLS = ("SimpleRNN", "LSTM", "GRU")


def eval_rnn(cell: str, cfg, texts: list[str], y_true: np.ndarray, labels: list[str]) -> dict | None:
    art = cfg.path("artifacts", cell)
    if not (art / "weights.pt").exists():
        print(f"skip {cell}: no weights at {art}")
        return None
    rc = cfg["rnn"]
    vec = WordVectorizer.load(art / "vectorizer.json")
    model = RecurrentClassifier(
        cell=cell,
        vocab_size=vec.effective_vocab_size,
        num_classes=len(labels),
        embed_dim=rc["embed_dim"],
        hidden_dim=rc["hidden_dim"],
        dense_dim=rc["dense_dim"],
        dropout=rc["dropout"],
    )
    model.load_state_dict(torch.load(art / "weights.pt", map_location="cpu"))
    model.eval()

    x = vec.transform(texts)
    probs = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            batch = torch.from_numpy(x[start : start + 256])
            probs.append(torch.softmax(model(batch), dim=-1).numpy())
    probs = np.concatenate(probs)

    def predict_one(text: str) -> int:
        with torch.no_grad():
            return int(model(torch.from_numpy(vec.transform([text]))).argmax(dim=-1).item())

    metrics = compute_metrics(y_true, probs.argmax(axis=1), labels)
    train_meta_path = cfg.path("results", "metrics", f"train_{cell}.json")
    train_meta = json.loads(train_meta_path.read_text()) if train_meta_path.exists() else {}
    metrics.update(
        {
            "model": cell,
            "trainable_parameters": train_meta.get("trainable_parameters"),
            "train_time_sec": train_meta.get("train_time_sec"),
            "train_device": train_meta.get("device"),
            "epochs_run": train_meta.get("epochs_run"),
            "best_val_f1_macro": train_meta.get("best_val_f1_macro"),
            "artifact_size_mb": directory_size_mb(art),
            "cpu_latency": measure_cpu_latency(predict_one, texts[:40]),
        }
    )
    np.save(art / "test_probs.npy", probs)
    return metrics


def eval_distilbert(cfg, texts: list[str], y_true: np.ndarray, labels: list[str]) -> dict | None:
    art = cfg.path("artifacts", "DistilBERT")
    if not (art / "config.json").exists():
        print(f"skip DistilBERT: no checkpoint at {art}")
        return None
    from src.inference.predictor import DistilBertPredictor

    predictor = DistilBertPredictor(art, cfg=cfg, device="cpu")
    probs = predictor.probabilities(texts, batch_size=cfg["distilbert"]["eval_batch_size"])
    metrics = compute_metrics(y_true, probs.argmax(axis=1), labels)
    train_meta_path = cfg.path("results", "metrics", "train_DistilBERT.json")
    train_meta = json.loads(train_meta_path.read_text()) if train_meta_path.exists() else {}
    metrics.update(
        {
            "model": "DistilBERT",
            "trainable_parameters": train_meta.get("trainable_parameters"),
            "train_time_sec": train_meta.get("train_time_sec"),
            "train_device": train_meta.get("device"),
            "epochs_run": train_meta.get("epochs_run"),
            "best_val_f1_macro": train_meta.get("best_val_f1_macro"),
            "artifact_size_mb": directory_size_mb(art),
            "cpu_latency": measure_cpu_latency(lambda t: predictor.predict(t).label, texts[:40]),
        }
    )
    np.save(art / "test_probs.npy", probs)
    return metrics


def main() -> int:
    cfg = load_config()
    set_seed(cfg["seed"])
    labels = list(cfg["data"]["labels"])
    test_df = load_split(cfg.path(cfg["data"]["processed_dir"]), "test")
    texts = test_df[cfg["data"]["text_column"]].tolist()
    y_true = test_df["label_id"].to_numpy(dtype=np.int64)
    print(f"test split: {len(texts):,} complaints")

    results: dict[str, dict] = {}
    for cell in RNN_CELLS:
        got = eval_rnn(cell, cfg, texts, y_true, labels)
        if got:
            results[cell] = got
            print(f"{cell:<11} acc={got['accuracy']:.4f} macroF1={got['f1_macro']:.4f}")
    got = eval_distilbert(cfg, texts, y_true, labels)
    if got:
        results["DistilBERT"] = got
        print(f"{'DistilBERT':<11} acc={got['accuracy']:.4f} macroF1={got['f1_macro']:.4f}")

    if not results:
        print("no trained models found", file=sys.stderr)
        return 1

    metrics_dir = cfg.path("results", "metrics")
    for name, payload in results.items():
        save_json(payload, metrics_dir / f"test_{name}.json")

    order = [m for m in (*RNN_CELLS, "DistilBERT") if m in results]
    table = pd.DataFrame(
        [
            {
                "model": m,
                "accuracy": round(results[m]["accuracy"], 4),
                "precision_macro": round(results[m]["precision_macro"], 4),
                "recall_macro": round(results[m]["recall_macro"], 4),
                "f1_macro": round(results[m]["f1_macro"], 4),
                "f1_weighted": round(results[m]["f1_weighted"], 4),
                "train_time_sec": results[m].get("train_time_sec"),
                "cpu_latency_p50_ms": round(results[m]["cpu_latency"]["p50_ms"], 2),
                "artifact_size_mb": results[m]["artifact_size_mb"],
                "trainable_parameters": results[m].get("trainable_parameters"),
            }
            for m in order
        ]
    )
    table.to_csv(metrics_dir / "model_comparison.csv", index=False)
    save_json(
        {
            "test_split_size": int(len(texts)),
            "labels": labels,
            "models": {m: results[m] for m in order},
        },
        metrics_dir / "model_comparison.json",
    )
    print("\n" + table.to_string(index=False))
    print(f"\nwrote {metrics_dir / 'model_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
