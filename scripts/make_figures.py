"""Render the static figures the README embeds, from the real metrics files.

    python scripts/make_figures.py

Reads results/metrics/*.json only. If a metric file is missing the figure is
skipped with a message rather than drawn from a placeholder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app_theme import BLUE, CATEGORY_COLORS, NAVY  # noqa: E402
from src.config import display_name, load_config  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.edgecolor": "#D5DDE8",
    "axes.labelcolor": "#334155",
    "axes.titlecolor": NAVY,
    "axes.titleweight": "semibold",
    "axes.grid": True,
    "grid.color": "#EEF2F7",
    "grid.linewidth": 0.8,
    "xtick.color": "#5B6B82",
    "ytick.color": "#5B6B82",
    "figure.facecolor": "white",
})
ORDER = ("SimpleRNN", "LSTM", "GRU", "DistilBERT")


def spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def main() -> int:
    cfg = load_config()
    metrics = cfg.path("results", "metrics")
    figures = cfg.path("results", "figures")
    figures.mkdir(parents=True, exist_ok=True)
    labels = [display_name(label, cfg) for label in cfg["data"]["labels"]]
    written: list[str] = []

    def load(name: str):
        path = metrics / name
        return json.loads(path.read_text()) if path.exists() else None

    # --- class distribution -------------------------------------------------
    quality = load("data_quality_report.json")
    if quality:
        shares = quality["label_shares"]
        names = [display_name(k, cfg) for k in shares]
        values = [v * 100 for v in shares.values()]
        order = np.argsort(values)[::-1]
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        bars = ax.bar([names[i] for i in order], [values[i] for i in order],
                      color=[CATEGORY_COLORS.get(names[i], BLUE) for i in order])
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
        ax.set_ylabel("Share of complaints")
        ax.set_title(f"Class distribution after cleaning ({quality['n_output']:,} complaints)")
        ax.set_ylim(0, max(values) * 1.18)
        plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
        spines(ax)
        fig.savefig(figures / "class_distribution.png")
        plt.close(fig)
        written.append("class_distribution.png")

    comparison = load("model_comparison.json")
    if not comparison:
        print("no model_comparison.json yet; model figures skipped")
        print("wrote:", ", ".join(written) or "nothing")
        return 0

    models = [m for m in ORDER if m in comparison["models"]]
    payloads = [comparison["models"][m] for m in models]

    # --- accuracy / macro F1 ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, key, title in zip(axes, ("accuracy", "f1_macro"), ("Accuracy", "Macro F1")):
        values = [p[key] for p in payloads]
        colours = [NAVY if m == models[int(np.argmax(values))] else BLUE for m in models]
        bars = ax.bar(models, values, color=colours)
        ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
        ax.set_ylim(min(values) - 0.06, 1.0)
        ax.set_title(f"{title} (held-out test, {comparison['test_split_size']:,})")
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
        spines(ax)
    fig.tight_layout()
    fig.savefig(figures / "model_comparison.png")
    plt.close(fig)
    written.append("model_comparison.png")

    # --- cost of accuracy ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for model, payload in zip(models, payloads):
        ax.scatter(payload["cpu_latency"]["p50_ms"], payload["artifact_size_mb"],
                   s=70 + payload["f1_macro"] * 260, color=BLUE,
                   edgecolor="white", linewidth=1.6, zorder=3)
        ax.annotate(f"{model}\n{payload['f1_macro']:.3f}",
                    (payload["cpu_latency"]["p50_ms"], payload["artifact_size_mb"]),
                    textcoords="offset points", xytext=(0, 14), ha="center", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("CPU latency p50 (ms)")
    ax.set_ylabel("Artifact size (MB)")
    ax.set_title("Cost of accuracy (bubble area = macro F1)")
    spines(ax)
    fig.savefig(figures / "latency_vs_size.png")
    plt.close(fig)
    written.append("latency_vs_size.png")

    # --- confusion matrices -------------------------------------------------
    cols = len(models)
    fig, axes = plt.subplots(1, cols, figsize=(4.4 * cols, 4.2), squeeze=False)
    for ax, model, payload in zip(axes[0], models, payloads):
        matrix = np.array(payload["confusion_matrix"], dtype=float)
        normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(f"{model}  (macro F1 {payload['f1_macro']:.4f})", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.grid(False)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{normalised[i, j]:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if normalised[i, j] > 0.5 else "#1E293B")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrices.png")
    plt.close(fig)
    written.append("confusion_matrices.png")

    # --- training curves ----------------------------------------------------
    histories = {m: load(f"train_{m}.json") for m in models}
    histories = {m: h for m, h in histories.items() if h and h.get("history")}
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
        for model, meta in histories.items():
            epochs = [h["epoch"] for h in meta["history"]]
            axes[0].plot(epochs, [h["val_loss"] for h in meta["history"]], marker="o",
                         markersize=3.5, label=model)
            axes[1].plot(epochs, [h["val_f1_macro"] for h in meta["history"]], marker="o",
                         markersize=3.5, label=model)
        axes[0].set_title("Validation loss"); axes[0].set_xlabel("Epoch")
        axes[1].set_title("Validation macro F1"); axes[1].set_xlabel("Epoch")
        for ax in axes:
            ax.legend(frameon=False, fontsize=8)
            spines(ax)
        fig.tight_layout()
        fig.savefig(figures / "training_curves.png")
        plt.close(fig)
        written.append("training_curves.png")

    # --- per-class F1 -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    width = 0.8 / len(models)
    positions = np.arange(len(labels))
    for i, (model, payload) in enumerate(zip(models, payloads)):
        values = [payload["per_class"][key]["f1"] for key in cfg["data"]["labels"]]
        ax.bar(positions + i * width, values, width, label=model)
    ax.set_xticks(positions + width * (len(models) - 1) / 2, labels, rotation=18, ha="right")
    ax.set_ylabel("F1"); ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 on the held-out test set")
    ax.legend(frameon=False, fontsize=8, ncols=len(models))
    spines(ax)
    fig.savefig(figures / "per_class_f1.png")
    plt.close(fig)
    written.append("per_class_f1.png")

    # --- threshold sweep ----------------------------------------------------
    threshold = load("review_threshold.json")
    if threshold and threshold.get("sweep"):
        sweep = [row for row in threshold["sweep"] if row["auto_accuracy"] is not None]
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot([r["threshold"] for r in sweep], [r["auto_accuracy"] for r in sweep],
                color=BLUE, label="accuracy on auto-routed slice")
        ax.plot([r["threshold"] for r in sweep], [r["coverage"] for r in sweep],
                color="#B45309", linestyle="--", label="share routed automatically")
        ax.axvline(threshold["threshold"], color=NAVY, linewidth=1.2)
        ax.annotate(f"chosen {threshold['threshold']:.2f}",
                    (threshold["threshold"], 0.5), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color=NAVY)
        ax.set_xlabel("Confidence threshold"); ax.set_ylim(0, 1.02)
        ax.set_title("Review threshold, chosen on the validation split")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        spines(ax)
        fig.savefig(figures / "review_threshold.png")
        plt.close(fig)
        written.append("review_threshold.png")

    print("wrote to results/figures/:", ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
