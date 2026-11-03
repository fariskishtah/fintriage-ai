"""Regenerate the README's results section from results/metrics/.

Numbers in the README are never typed by hand: this script renders them from the
metrics files, between HTML comment markers, so the README cannot drift from the
experiments or report a figure no run produced.

    python scripts/update_readme.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import display_name, load_config  # noqa: E402

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
ORDER = ("SimpleRNN", "LSTM", "GRU", "DistilBERT")
REFERENCE = {  # the original author's published numbers, shown as reference only
    "SimpleRNN": 0.7731, "LSTM": 0.8122, "GRU": 0.8257, "DistilBERT": 0.8433,
}


def main() -> int:
    cfg = load_config()
    metrics = cfg.path("results", "metrics")
    figures_rel = "results/figures"

    def load(name: str):
        path = metrics / name
        return json.loads(path.read_text()) if path.exists() else None

    comparison = load("model_comparison.json")
    selection = load("model_selection.json")
    threshold = load("review_threshold.json")

    if not comparison:
        print("results/metrics/model_comparison.json missing — run scripts/evaluate.py first",
              file=sys.stderr)
        return 1

    models = [m for m in ORDER if m in comparison["models"]]
    winner = selection["selected_model"] if selection else None
    lines: list[str] = []

    lines.append("## Results")
    lines.append("")
    lines.append(
        f"All four models evaluated on the same held-out test split of "
        f"**{comparison['test_split_size']:,} complaints**, never seen during "
        f"training or model selection. Every number below came from our own runs."
    )
    lines.append("")

    # --- headline table ----------------------------------------------------
    lines.append("| Model | Accuracy | Precision (macro) | Recall (macro) | **Macro F1** | Weighted F1 | Train time | CPU latency p50 | Size | Params |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in models:
        p = comparison["models"][name]
        label = f"**{name}**" if name == winner else name
        f1 = f"**{p['f1_macro']:.4f}**" if name == winner else f"{p['f1_macro']:.4f}"
        train_time = f"{p['train_time_sec']:,.0f} s" if p.get("train_time_sec") else "—"
        params = f"{p['trainable_parameters']:,}" if p.get("trainable_parameters") else "—"
        lines.append(
            f"| {label} | {p['accuracy']:.4f} | {p['precision_macro']:.4f} | "
            f"{p['recall_macro']:.4f} | {f1} | {p['f1_weighted']:.4f} | {train_time} | "
            f"{p['cpu_latency']['p50_ms']:.0f} ms | {p['artifact_size_mb']:.0f} MB | {params} |"
        )
    lines.append("")

    if winner:
        p = comparison["models"][winner]
        lines.append(
            f"**Production model: {winner}** — selected on validation macro F1 "
            f"before the test split was read. "
            f"{selection.get('decision_basis', '')}."
        )
        lines.append("")

    # --- comparison with the reference project -----------------------------
    lines.append("### Against the project that inspired this one")
    lines.append("")
    lines.append(
        "The reference implementation reports the macro F1 values below. Ours are "
        "not expected to match: we remove near-duplicate narratives across splits, "
        "which that protocol does not, so a slice of what it scores is memorisation "
        "rather than generalisation. The comparison is here for honesty, not as a claim."
    )
    lines.append("")
    lines.append("| Model | Reference (published) | Ours | Difference |")
    lines.append("|---|---|---|---|")
    for name in models:
        ours = comparison["models"][name]["f1_macro"]
        ref = REFERENCE.get(name)
        if ref is None:
            continue
        delta = ours - ref
        lines.append(f"| {name} | {ref:.4f} | {ours:.4f} | {delta:+.4f} |")
    lines.append("")

    # --- per class ---------------------------------------------------------
    if winner:
        lines.append(f"### Per-class performance — {winner}")
        lines.append("")
        lines.append("| Category | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for label in cfg["data"]["labels"]:
            s = comparison["models"][winner]["per_class"][label]
            lines.append(
                f"| {display_name(label, cfg)} | {s['precision']:.4f} | "
                f"{s['recall']:.4f} | {s['f1']:.4f} | {s['support']:,} |"
            )
        lines.append("")

    # --- routing threshold --------------------------------------------------
    if threshold:
        lines.append("### Human-review threshold")
        lines.append("")
        auto_acc = threshold.get("achieved_auto_accuracy")
        lines.append(
            f"Chosen on the **validation** split, not the test split: "
            f"**{threshold['threshold']:.2f}**, the "
            f"{threshold.get('selection_rule', 'selected threshold')}."
        )
        lines.append("")
        lines.append("| | |")
        lines.append("|---|---|")
        lines.append(f"| Routed automatically | {threshold.get('auto_coverage', 0):.1%} of complaints |")
        if auto_acc is not None:
            lines.append(f"| Accuracy on that slice | {auto_acc:.1%} |")
        lines.append(f"| Sent to manual review | {threshold.get('review_coverage', 0):.1%} |")
        lines.append(f"| Expected calibration error | {threshold.get('expected_calibration_error')} |")
        lines.append("")
        lines.append(
            "> Confidence values are raw softmax outputs, **not calibrated "
            "probabilities**. The ECE above is how far they sit from observed "
            "accuracy. The app presents confidence as a routing signal, never as a "
            "guarantee of correctness."
        )
        lines.append("")

    # --- figures ------------------------------------------------------------
    available = [
        ("model_comparison.png", "Accuracy and macro F1 across the four architectures"),
        ("confusion_matrices.png", "Confusion matrices, row-normalised"),
        ("per_class_f1.png", "Per-class F1"),
        ("training_curves.png", "Validation loss and macro F1 per epoch"),
        ("latency_vs_size.png", "Latency against model size"),
        ("review_threshold.png", "Threshold sweep on the validation split"),
        ("class_distribution.png", "Class distribution after cleaning"),
    ]
    present = [(f, c) for f, c in available if cfg.path("results", "figures", f).exists()]
    if present:
        lines.append("### Figures")
        lines.append("")
        for filename, caption in present:
            lines.append(f"**{caption}**")
            lines.append("")
            lines.append(f"![{caption}]({figures_rel}/{filename})")
            lines.append("")

    block = "\n".join(lines).rstrip()
    readme = cfg.path("README.md")
    text = readme.read_text(encoding="utf-8")

    if START in text and END in text:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        text = f"{head}{START}\n\n{block}\n\n{END}{tail}"
    else:
        print(f"markers not found in README.md; append {START} / {END} where the "
              "results should go", file=sys.stderr)
        return 1

    # Drop the "results pending" warning once there are real results.
    pending = (
        "> ⚠️ **Results pending.** This README is updated with real measured metrics once\n"
        "> all four models finish training. Nothing in this repository reports a metric we\n"
        "> have not produced ourselves. Current state: `PROJECT_STATUS.md`.\n\n"
    )
    text = text.replace(pending, "")
    readme.write_text(text, encoding="utf-8")
    print(f"README results section rewritten from {len(models)} model(s)"
          + (f", production model {winner}" if winner else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
