"""Train one recurrent model (SimpleRNN | LSTM | GRU) and record real metrics.

    python scripts/train_rnn.py --cell GRU

Writes artifacts/<cell>/{weights.pt,vectorizer.json,history.json} and
results/metrics/<cell>.json. Validation drives early stopping and the saved
checkpoint; the test split is only touched by scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, set_seed  # noqa: E402
from src.evaluation.metrics import compute_metrics, save_json  # noqa: E402
from src.models.rnn import CELLS, RecurrentClassifier  # noqa: E402
from src.preprocessing.vectorizer import WordVectorizer  # noqa: E402
from src.training.common import (  # noqa: E402
    EarlyStopping,
    balanced_class_weights,
    load_split,
    pick_device,
)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    for xb, _ in loader:
        chunks.append(model(xb.to(device)).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="train on N examples only (smoke run)")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    labels = list(cfg["data"]["labels"])
    rc, data = cfg["rnn"], cfg["data"]
    text_col = data["text_column"]
    device = pick_device(args.device)
    epochs = args.epochs or rc["epochs"]
    print(f"cell={args.cell} device={device} epochs={epochs}")

    proc = cfg.path(data["processed_dir"])
    train_df = load_split(proc, "train")
    val_df = load_split(proc, "val")
    if args.limit:
        train_df = train_df.sample(n=min(args.limit, len(train_df)), random_state=cfg["seed"])
        val_df = val_df.sample(n=min(max(args.limit // 4, 1), len(val_df)), random_state=cfg["seed"])
        epochs = 1
        print(f"SMOKE RUN: {len(train_df)} train / {len(val_df)} val, 1 epoch")

    vec = WordVectorizer(vocab_size=rc["vocab_size"], max_len=rc["max_len"])
    x_train = vec.fit_transform(train_df[text_col].tolist())  # fitted on train only
    x_val = vec.transform(val_df[text_col].tolist())
    y_train = train_df["label_id"].to_numpy(dtype=np.int64)
    y_val = val_df["label_id"].to_numpy(dtype=np.int64)
    print(f"vocab={vec.effective_vocab_size:,} train={len(y_train):,} val={len(y_val):,}")

    model = RecurrentClassifier(
        cell=args.cell,
        vocab_size=vec.effective_vocab_size,
        num_classes=len(labels),
        embed_dim=rc["embed_dim"],
        hidden_dim=rc["hidden_dim"],
        dense_dim=rc["dense_dim"],
        dropout=rc["dropout"],
    ).to(device)
    n_params = model.num_trainable_parameters()
    print(f"trainable params: {n_params:,}")

    weights = balanced_class_weights(y_train, len(labels))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=rc["lr"])

    train_loader = make_loader(x_train, y_train, rc["batch_size"], shuffle=True)
    val_loader = make_loader(x_val, y_val, rc["batch_size"] * 2, shuffle=False)

    out_dir = cfg.path("artifacts", args.cell)
    out_dir.mkdir(parents=True, exist_ok=True)
    stopper = EarlyStopping(patience=rc["early_stopping_patience"], mode="max")
    history: list[dict] = []
    wall_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), rc["grad_clip"])
            optimiser.step()
            running += loss.item() * len(yb)
            seen += len(yb)
        train_loss = running / max(seen, 1)

        val_logits = predict_logits(model, val_loader, device)
        val_loss = float(
            criterion(
                torch.from_numpy(val_logits).to(device), torch.from_numpy(y_val).to(device)
            ).item()
        )
        val_pred = val_logits.argmax(axis=1)
        val_metrics = compute_metrics(y_val, val_pred, labels)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val_metrics["accuracy"], 5),
            "val_f1_macro": round(val_metrics["f1_macro"], 5),
        }
        history.append(row)
        print(json.dumps(row))

        improved, stop = stopper.step(val_metrics["f1_macro"])
        if improved:
            torch.save(model.state_dict(), out_dir / "weights.pt")
            vec.save(out_dir / "vectorizer.json")
            np.save(out_dir / "val_probs.npy", torch.softmax(torch.from_numpy(val_logits), 1).numpy())
        if stop:
            print(f"early stopping after epoch {epoch} (best val macro-F1 {stopper.best:.4f})")
            break

    train_seconds = time.perf_counter() - wall_start
    meta = {
        "model": args.cell,
        "device": str(device),
        "trainable_parameters": n_params,
        "train_time_sec": round(train_seconds, 1),
        "epochs_run": len(history),
        "best_val_f1_macro": round(stopper.best, 5),
        "class_weights": {labels[i]: round(float(w), 4) for i, w in enumerate(weights)},
        "hyperparameters": dict(rc),
        "history": history,
    }
    meta["smoke_run"] = bool(args.limit)
    save_json(meta, out_dir / "history.json")
    if not args.limit:
        save_json(meta, cfg.path("results", "metrics", f"train_{args.cell}.json"))
    print(f"done in {train_seconds:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
