"""Fine-tune distilbert-base-uncased with a class-weighted loss.

    python scripts/train_distilbert.py                 # full run
    python scripts/train_distilbert.py --smoke 2000    # cheap wiring check first

Uses a plain PyTorch loop rather than HF Trainer: the weighted loss, the
step-level checkpointing and the MPS/CPU fallbacks are all easier to see and to
resume from. Checkpoints are written every --save-every steps so an interrupted
run resumes instead of restarting.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, set_seed  # noqa: E402
from src.evaluation.metrics import compute_metrics, save_json  # noqa: E402
from src.training.common import balanced_class_weights, load_split, pick_device  # noqa: E402


class ComplaintDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_len: int) -> None:
        self.texts, self.labels = texts, labels
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs, trues, loss_sum, seen = [], [], 0.0, 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        loss_sum += criterion(logits, labels).item() * len(labels)
        seen += len(labels)
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        trues.append(labels.cpu().numpy())
    return (
        np.concatenate(probs),
        np.concatenate(trues),
        loss_sum / max(seen, 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0, help="train on N examples only")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    cfg = load_config()
    set_seed(cfg["seed"])
    labels = list(cfg["data"]["labels"])
    db, data = cfg["distilbert"], cfg["data"]
    text_col = data["text_column"]
    device = pick_device(args.device)
    epochs = args.epochs or db["epochs"]

    proc = cfg.path(data["processed_dir"])
    train_df, val_df = load_split(proc, "train"), load_split(proc, "val")
    if args.smoke:
        train_df = train_df.sample(n=min(args.smoke, len(train_df)), random_state=cfg["seed"])
        val_df = val_df.sample(n=min(args.smoke // 4 or 1, len(val_df)), random_state=cfg["seed"])
        epochs = 1
        print(f"SMOKE RUN: {len(train_df)} train / {len(val_df)} val, 1 epoch")

    tok = AutoTokenizer.from_pretrained(db["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        db["model_name"],
        num_labels=len(labels),
        id2label={i: n for i, n in enumerate(labels)},
        label2id={n: i for i, n in enumerate(labels)},
    ).to(device)

    y_train = train_df["label_id"].to_numpy(dtype=np.int64)
    y_val = val_df["label_id"].to_numpy(dtype=np.int64)
    weights = balanced_class_weights(y_train, len(labels))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    train_loader = DataLoader(
        ComplaintDataset(train_df[text_col].tolist(), y_train, tok, db["max_len"]),
        batch_size=db["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        ComplaintDataset(val_df[text_col].tolist(), y_val, tok, db["max_len"]),
        batch_size=db["eval_batch_size"],
        shuffle=False,
    )

    total_steps = len(train_loader) * epochs
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=db["lr"], weight_decay=db["weight_decay"]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(total_steps * db["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    out_dir = cfg.path("artifacts", "DistilBERT")
    ckpt_dir = cfg.path("artifacts", "checkpoints", "DistilBERT")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / "trainer_state.pt"

    start_epoch, global_step, best_f1 = 1, 0, float("-inf")
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, global_step, best_f1 = state["epoch"], state["global_step"], state["best_f1"]
        print(f"resumed from epoch {start_epoch}, step {global_step}, best F1 {best_f1:.4f}")

    def save_state(epoch: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_f1": best_f1,
            },
            state_path,
        )

    history: list[dict] = []
    wall_start = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            batch_labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(**batch).logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), db["grad_clip"])
            optimiser.step()
            scheduler.step()
            global_step += 1
            running += loss.item() * len(batch_labels)
            seen += len(batch_labels)
            if step % 50 == 0:
                done = step / len(train_loader)
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"({done:5.1%}) loss {running / seen:.4f}",
                    flush=True,
                )
            if args.save_every and global_step % args.save_every == 0:
                save_state(epoch)

        train_loss = running / max(seen, 1)
        probs, trues, val_loss = evaluate(model, val_loader, device, criterion)
        val_metrics = compute_metrics(trues, probs.argmax(axis=1), labels)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val_metrics["accuracy"], 5),
            "val_f1_macro": round(val_metrics["f1_macro"], 5),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            np.save(out_dir / "val_probs.npy", probs)
            np.save(out_dir / "val_labels.npy", trues)
            print(f"  saved best model (val macro-F1 {best_f1:.4f}) -> {out_dir}")
        save_state(epoch + 1)

    train_seconds = time.perf_counter() - wall_start
    meta = {
        "model": "DistilBERT",
        "base_checkpoint": db["model_name"],
        "device": str(device),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_time_sec": round(train_seconds, 1),
        "epochs_run": len(history),
        "best_val_f1_macro": round(best_f1, 5),
        "class_weights": {labels[i]: round(float(w), 4) for i, w in enumerate(weights)},
        "hyperparameters": dict(db),
        "smoke_run": bool(args.smoke),
        "history": history,
    }
    save_json(meta, out_dir / "history.json")
    if not args.smoke:
        save_json(meta, cfg.path("results", "metrics", "train_DistilBERT.json"))
    print(f"done in {train_seconds / 60:.1f} min -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
