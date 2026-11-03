"""Single entry point for predictions, shared by the app, the tests and the
evaluation script, so there is exactly one place where text becomes a label."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import Config, display_name, load_config


@dataclass
class Prediction:
    label: str
    display: str
    confidence: float
    probabilities: dict[str, float]
    needs_review: bool
    threshold: float

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        ordered = sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return ordered[:k]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "display": self.display,
            "confidence": round(self.confidence, 6),
            "needs_review": self.needs_review,
            "threshold": self.threshold,
            "probabilities": {k: round(v, 6) for k, v in self.probabilities.items()},
        }


def load_review_threshold(cfg: Config) -> float:
    """Prefer the threshold chosen on the validation split; fall back to config."""
    path = cfg.path("results", "metrics", "review_threshold.json")
    if path.exists():
        return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
    return float(cfg["inference"]["review_threshold"])


class DistilBertPredictor:
    """Wraps the fine-tuned transformer. CPU by default: that is what the
    deployed Space runs on, and what the latency numbers are measured on."""

    def __init__(
        self,
        model_dir: str | Path,
        cfg: Config | None = None,
        device: str = "cpu",
        threshold: float | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.cfg = cfg or load_config()
        self.labels = list(self.cfg["data"]["labels"])
        self.device = torch.device(device)
        self.max_len = int(self.cfg["distilbert"]["max_len"])
        self.model_dir = Path(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device).eval()

        # The checkpoint carries its own id2label; refuse to run if it disagrees
        # with the project label order rather than silently mislabelling output.
        ckpt_labels = [
            self.model.config.id2label[i] for i in range(self.model.config.num_labels)
        ]
        if ckpt_labels != self.labels:
            raise ValueError(
                f"label order mismatch.\n  checkpoint: {ckpt_labels}\n  config:     {self.labels}"
            )
        self.threshold = threshold if threshold is not None else load_review_threshold(self.cfg)
        self._torch = torch

    def probabilities(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = [str(t) for t in texts[start : start + batch_size]]
                enc = self.tokenizer(
                    chunk,
                    truncation=True,
                    max_length=self.max_len,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, len(self.labels)))

    def predict(self, text: str) -> Prediction:
        if not str(text).strip():
            raise ValueError("empty complaint text")
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[Prediction]:
        probs = self.probabilities(texts, batch_size=batch_size)
        results = []
        for row in probs:
            idx = int(row.argmax())
            label = self.labels[idx]
            confidence = float(row[idx])
            results.append(
                Prediction(
                    label=label,
                    display=display_name(label, self.cfg),
                    confidence=confidence,
                    probabilities={
                        name: float(row[i]) for i, name in enumerate(self.labels)
                    },
                    needs_review=confidence < self.threshold,
                    threshold=self.threshold,
                )
            )
        return results


def resolve_model_dir(cfg: Config | None = None) -> Path:
    """Local artifacts first, then FINTRIAGE_MODEL_DIR, then the HF repo."""
    import os

    cfg = cfg or load_config()
    env_dir = os.environ.get("FINTRIAGE_MODEL_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    local = cfg.path("artifacts", "DistilBERT")
    if (local / "config.json").exists():
        return local
    repo = os.environ.get("FINTRIAGE_MODEL_REPO")
    if repo:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=repo))
    raise FileNotFoundError(
        "no model found. Train it (scripts/train_distilbert.py), or set "
        "FINTRIAGE_MODEL_DIR / FINTRIAGE_MODEL_REPO."
    )
