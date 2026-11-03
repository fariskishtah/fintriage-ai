"""Deploy the app to a Hugging Face Space, and the checkpoint to a model repo.

    export HF_TOKEN=hf_...            # write scope
    python scripts/deploy_space.py --model-repo <user>/fintriage-distilbert \
                                   --space <user>/fintriage-ai

Weights go to the model repo (268 MB has no business in a Space's git history);
the Space pulls them at startup via FINTRIAGE_MODEL_REPO. Nothing is made public
unless --public is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Exactly what the running app needs. No data, no notebooks, no tests.
SPACE_FILES = [
    "app.py",
    "configs/config.yaml",
    "src/__init__.py",
    "src/config.py",
    "src/app_theme.py",
    "src/ui_blocks.py",
    "src/charts.py",
    "src/inference/__init__.py",
    "src/inference/predictor.py",
    "src/inference/batch.py",
    "src/inference/examples.py",
    "src/explainability/__init__.py",
    "src/explainability/integrated_gradients.py",
    "src/preprocessing/__init__.py",
    "src/preprocessing/quality.py",
]
# Metrics the Model Performance tab reads. Small JSON, no narratives.
SPACE_METRICS = [
    "model_comparison.json", "model_selection.json", "review_threshold.json",
    "data_quality_report.json", "split_manifest.json", "dataset_provenance.json",
    "label_map.json", "train_SimpleRNN.json", "train_LSTM.json", "train_GRU.json",
    "train_DistilBERT.json",
]


def stage_space(model_repo: str) -> Path:
    staging = Path(tempfile.mkdtemp(prefix="fintriage-space-"))
    for rel in SPACE_FILES:
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"missing {rel}")
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    metrics_dir = staging / "results" / "metrics"
    metrics_dir.mkdir(parents=True)
    copied = []
    for name in SPACE_METRICS:
        src = ROOT / "results" / "metrics" / name
        if src.exists():
            shutil.copy(src, metrics_dir / name)
            copied.append(name)
    print(f"metrics included: {len(copied)}")

    shutil.copy(ROOT / "deploy" / "README_SPACE.md", staging / "README.md")
    shutil.copy(ROOT / "deploy" / "requirements-space.txt", staging / "requirements.txt")
    # The Space resolves weights from the model repo at startup.
    (staging / ".env").write_text(f"FINTRIAGE_MODEL_REPO={model_repo}\n")
    (staging / "app_config.py").write_text(
        "import os\n"
        f'os.environ.setdefault("FINTRIAGE_MODEL_REPO", "{model_repo}")\n'
    )
    # Prepend the env default so predictor.resolve_model_dir finds the repo.
    app = staging / "app.py"
    app.write_text("import app_config  # noqa: F401  (sets FINTRIAGE_MODEL_REPO)\n"
                   + app.read_text())

    for path in staging.rglob("*"):
        if path.is_file() and (path.suffix in {".pem", ".key"} or path.name == "kaggle.json"):
            raise SystemExit(f"refusing to upload {path}")
    size_kb = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file()) / 1024
    print(f"staged {size_kb:.0f} KB for the Space")
    return staging


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="e.g. fariskishtah/fintriage-ai")
    ap.add_argument("--model-repo", required=True, help="e.g. fariskishtah/fintriage-distilbert")
    ap.add_argument("--public", action="store_true", help="make both repos public")
    ap.add_argument("--skip-weights", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set.\n"
            "  1. https://huggingface.co/settings/tokens -> New token, type 'Write'\n"
            "  2. export HF_TOKEN=hf_...\n"
            "  3. re-run this command"
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()
    print(f"hugging face user: {who['name']}")
    private = not args.public
    print(f"visibility: {'public' if args.public else 'private'}")

    checkpoint = ROOT / "artifacts" / "DistilBERT"
    if not args.skip_weights:
        if not (checkpoint / "model.safetensors").exists():
            raise SystemExit(
                f"no checkpoint at {checkpoint}. Train it first "
                "(notebooks/kaggle_train_distilbert.ipynb), or pass --skip-weights."
            )
        api.create_repo(args.model_repo, repo_type="model", exist_ok=True, private=private)
        api.upload_folder(
            folder_path=str(checkpoint),
            repo_id=args.model_repo,
            repo_type="model",
            ignore_patterns=["*.npy", "history.json"],
        )
        print(f"weights -> https://huggingface.co/{args.model_repo}")

    staging = stage_space(args.model_repo)
    api.create_repo(args.space, repo_type="space", space_sdk="gradio",
                    exist_ok=True, private=private)
    api.upload_folder(folder_path=str(staging), repo_id=args.space, repo_type="space")
    url = f"https://huggingface.co/spaces/{args.space}"
    print(f"\nspace -> {url}")
    print("It builds for a few minutes. Verify at the URL above before calling it deployed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
