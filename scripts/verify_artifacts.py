"""Verify Kaggle-produced artifacts before they are allowed into the project.

The Kaggle run trains all four models and evaluates them in one session. This
script proves those artifacts are comparable with our frozen protocol, and only
then promotes them.

    python scripts/verify_artifacts.py            # checks only, changes nothing
    python scripts/verify_artifacts.py --promote  # adopt them after the checks pass

Why a gate at all:

* The Kaggle run also trains its own SimpleRNN/LSTM/GRU. Extracting straight
  over artifacts/ would silently replace the local runs and put training times
  from an M2 MPS backend beside times from a Tesla T4 in one table.
* The held-out test set must be read exactly once, by one coherent set of
  checkpoints. Evaluating a second set after seeing the first set's test numbers
  would turn the test split into a selection surface.

So: one source of truth for the comparison (the Kaggle session), and the local
runs kept beside it as a validation-only reproducibility check.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402

STAGING = "artifacts_kaggle"
LOCAL_BACKUP = "artifacts_local_mps"
MODELS = ("SimpleRNN", "LSTM", "GRU", "DistilBERT")


class Check:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def require(self, ok: bool, label: str, detail: str = "") -> bool:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)
        return ok

    def note(self, label: str, detail: str = "") -> None:
        print(f"  [note] {label}" + (f"  {detail}" if detail else ""))
        self.warnings.append(label)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    root = cfg.path()
    staging = root / STAGING
    chk = Check()

    if not staging.exists():
        print(f"no {STAGING}/ — run `python scripts/kaggle_launch.py --pull` first",
              file=sys.stderr)
        return 1

    k_metrics = staging / "results" / "metrics"
    k_artifacts = staging / "artifacts"

    # ---------------------------------------------------------------- splits
    print("\n1. Same frozen splits?")
    local_fp = load_json(cfg.path("results", "metrics", "split_fingerprint.json"))
    kaggle_fp = load_json(k_metrics / "split_fingerprint.json")
    if chk.require(bool(local_fp and kaggle_fp), "split fingerprints present on both sides"):
        for name in ("train", "val", "test"):
            a = local_fp["splits"][name]["content_sha256"]
            b = kaggle_fp["splits"][name]["content_sha256"]
            chk.require(a == b, f"{name} split identical", f"{a[:16]}…")
            chk.require(
                local_fp["splits"][name]["n_rows"] == kaggle_fp["splits"][name]["n_rows"],
                f"{name} row count identical",
                f"{local_fp['splits'][name]['n_rows']:,}",
            )
        chk.require(local_fp["seed"] == kaggle_fp["seed"], "seed identical",
                    str(local_fp["seed"]))
        chk.require(
            local_fp.get("dataset_sha256") == kaggle_fp.get("dataset_sha256"),
            "source dataset identical",
            str(local_fp.get("dataset_sha256", ""))[:16] + "…",
        )
        overlap = kaggle_fp.get("cross_split_text_overlap", {})
        chk.require(
            sum(overlap.values()) == 0,
            "no narrative shared across splits on the Kaggle side",
            str(overlap),
        )
        chk.require(
            kaggle_fp.get("duplicate_groups_spanning_splits") == 0,
            "no duplicate group spans splits",
        )

    # ----------------------------------------------------------- label order
    print("\n2. Same label mapping?")
    local_map = load_json(cfg.path("results", "metrics", "label_map.json"))
    kaggle_map = load_json(k_metrics / "label_map.json")
    chk.require(local_map == kaggle_map, "label_map.json identical", str(local_map))
    chk.require(
        list(cfg["data"]["labels"]) == [k for k, _ in sorted((local_map or {}).items(),
                                                            key=lambda kv: kv[1])],
        "config label order matches the mapping",
    )

    # ------------------------------------------------------------ checkpoints
    print("\n3. All four checkpoints present and loadable?")
    import torch

    from src.models.rnn import RecurrentClassifier
    from src.preprocessing.vectorizer import WordVectorizer

    rc = cfg["rnn"]
    for name in ("SimpleRNN", "LSTM", "GRU"):
        d = k_artifacts / name
        if not chk.require((d / "weights.pt").exists(), f"{name} weights present"):
            continue
        try:
            vec = WordVectorizer.load(d / "vectorizer.json")
            model = RecurrentClassifier(
                cell=name, vocab_size=vec.effective_vocab_size, num_classes=5,
                embed_dim=rc["embed_dim"], hidden_dim=rc["hidden_dim"],
                dense_dim=rc["dense_dim"], dropout=rc["dropout"],
            )
            model.load_state_dict(torch.load(d / "weights.pt", map_location="cpu"))
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(torch.from_numpy(
                    vec.transform(["the bank charged me an overdraft fee twice"]))), -1)
            chk.require(abs(float(probs.sum()) - 1.0) < 1e-4, f"{name} produces a distribution",
                        f"{model.num_trainable_parameters():,} params")
        except Exception as exc:  # noqa: BLE001
            chk.require(False, f"{name} loads", f"{type(exc).__name__}: {exc}")

    db = k_artifacts / "DistilBERT"
    if chk.require((db / "model.safetensors").exists(), "DistilBERT weights present"):
        try:
            from transformers import AutoConfig

            conf = AutoConfig.from_pretrained(db)
            ckpt_labels = [conf.id2label[i] for i in range(conf.num_labels)]
            chk.require(ckpt_labels == list(cfg["data"]["labels"]),
                        "DistilBERT label order matches the config", str(ckpt_labels))
            chk.require((db / "tokenizer.json").exists() or
                        (db / "vocab.txt").exists(), "tokenizer saved alongside")
        except Exception as exc:  # noqa: BLE001
            chk.require(False, "DistilBERT config readable", f"{type(exc).__name__}: {exc}")

    # ------------------------------------- selection used validation, not test
    print("\n4. Was the production model chosen on validation only?")
    selection = load_json(k_metrics / "model_selection.json")
    threshold = load_json(k_metrics / "review_threshold.json")
    if chk.require(bool(selection), "model_selection.json present"):
        basis = (selection.get("decision_basis") or "").lower()
        chk.require("validation" in basis, "selection basis names validation", basis)
        chk.require(
            all("val_f1_macro" in row for row in selection.get("ranking", [])),
            "ranking carries validation macro F1 for every candidate",
        )
        chk.require(
            not any("test" in str(k).lower() for row in selection.get("ranking", []) for k in row),
            "no test metric appears in the ranking",
        )
        print(f"       selected: {selection.get('selected_model')}")
    if threshold:
        chk.require(threshold.get("n_val", 0) > 0 or threshold.get("fell_back_to_config"),
                    "review threshold calibrated on the validation split",
                    f"threshold={threshold.get('threshold')}")

    # --------------------------------------------- test evaluated exactly once
    print("\n5. Test set read exactly once?")
    local_tests = list(cfg.path("results", "metrics").glob("test_*.json"))
    chk.require(not local_tests, "no prior local test evaluation",
                f"{len(local_tests)} found")
    kaggle_tests = sorted(p.name for p in k_metrics.glob("test_*.json"))
    chk.require(len(kaggle_tests) >= 1, "Kaggle produced test metrics", str(kaggle_tests))
    comparison = load_json(k_metrics / "model_comparison.json")
    if chk.require(bool(comparison), "model_comparison.json present"):
        chk.require(
            comparison["test_split_size"] == local_fp["splits"]["test"]["n_rows"],
            "comparison used the full frozen test split",
            f"{comparison['test_split_size']:,}",
        )
        chk.require(
            all(m in comparison["models"] for m in MODELS),
            "all four models in the comparison",
            str(list(comparison["models"])),
        )

    # ------------------------------- reproducibility: local vs Kaggle (val only)
    print("\n6. Reproducibility — local MPS vs Kaggle GPU (validation only)")
    backup = root / LOCAL_BACKUP / "metrics"
    if backup.exists():
        for name in ("SimpleRNN", "LSTM", "GRU"):
            local_meta = load_json(backup / f"train_{name}.json")
            kaggle_meta = load_json(k_metrics / f"train_{name}.json")
            if not (local_meta and kaggle_meta):
                continue
            a = local_meta["best_val_f1_macro"]
            b = kaggle_meta["best_val_f1_macro"]
            delta = b - a
            flag = "close" if abs(delta) < 0.03 else "DIVERGENT"
            print(f"  {name:<11} MPS {a:.4f}  |  T4 {b:.4f}  |  Δ {delta:+.4f}  {flag}")
            if abs(delta) >= 0.03:
                chk.note(f"{name} differs by {delta:+.4f} between hardware")
    else:
        chk.note("no preserved local run to compare against")

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 68)
    if chk.failures:
        print(f"VERIFICATION FAILED — {len(chk.failures)} check(s):")
        for f in chk.failures:
            print(f"  - {f}")
        print("\nNothing was promoted.")
        return 1
    print("VERIFICATION PASSED")
    print(
        "\nThe comparison will use the Kaggle session's four checkpoints only: all\n"
        "trained and evaluated on one machine, with one set of library versions, so\n"
        f"training times and CPU latencies are comparable. The local MPS runs stay in\n"
        f"{LOCAL_BACKUP}/ as a validation-only reproducibility record and are never\n"
        "mixed into the table."
    )
    if chk.warnings:
        print(f"\nNotes ({len(chk.warnings)}):")
        for w in chk.warnings:
            print(f"  - {w}")

    if not args.promote:
        print("\nRun again with --promote to adopt these artifacts.")
        return 0

    # ------------------------------------------------------------------ promote
    print("\nPromoting…")
    for name in MODELS:
        src = k_artifacts / name
        if not src.exists():
            continue
        dst = root / "artifacts" / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  artifacts/{name}")
    dst_metrics = cfg.path("results", "metrics")
    for src in sorted(k_metrics.glob("*.json")):
        shutil.copy(src, dst_metrics / src.name)
        print(f"  results/metrics/{src.name}")
    for src in sorted(k_metrics.glob("*.csv")):
        shutil.copy(src, dst_metrics / src.name)
        print(f"  results/metrics/{src.name}")
    provenance = {
        "comparison_source": "Kaggle kernel fariskishtah/fintriage-distilbert-training",
        "why": (
            "All four models trained and evaluated in one GPU session so training "
            "times and CPU latencies are measured on the same machine."
        ),
        "local_mps_runs_preserved_at": LOCAL_BACKUP,
        "local_runs_used_for": "validation-only reproducibility check; never in the table",
        "test_set_evaluations": 1,
        "split_fingerprints_verified": True,
    }
    (dst_metrics / "artifact_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("  results/metrics/artifact_provenance.json")
    print("\nPromoted. Next: scripts/make_figures.py, scripts/update_readme.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
