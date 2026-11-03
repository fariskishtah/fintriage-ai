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


def stage_space(model_repo: str, zerogpu: bool = False, bundle_weights: bool = False) -> Path:
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

    readme = "README_SPACE_ZEROGPU.md" if zerogpu else "README_SPACE.md"
    reqs = "requirements-zerogpu.txt" if zerogpu else "requirements-space.txt"
    shutil.copy(ROOT / "deploy" / readme, staging / "README.md")
    shutil.copy(ROOT / "deploy" / reqs, staging / "requirements.txt")

    if bundle_weights:
        # Ship the checkpoint inside the Space. The Space is private, so the
        # weights stay private, and nothing needs a token at runtime - which is
        # better than handing the Space a credential just to read our own repo.
        src = ROOT / "artifacts" / "DistilBERT"
        if not (src / "model.safetensors").exists():
            raise SystemExit(f"no checkpoint at {src}")
        shutil.copytree(src, staging / "model",
                        ignore=shutil.ignore_patterns("*.npy", "history.json"))
        config_line = 'os.environ.setdefault("FINTRIAGE_MODEL_DIR", str(HERE / "model"))\n'
        print(f"bundling weights into the Space: "
              f"{sum(f.stat().st_size for f in (staging / 'model').rglob('*') if f.is_file())/1e6:.0f} MB")
    else:
        config_line = f'os.environ.setdefault("FINTRIAGE_MODEL_REPO", "{model_repo}")\n'

    prelude = "" if not zerogpu else "import zerogpu_runtime  # noqa: F401\n"
    (staging / "app_config.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        + config_line
        + prelude
    )
    if zerogpu:
        shutil.copy(ROOT / "deploy" / "zerogpu_runtime.py", staging / "zerogpu_runtime.py")
    # Make the app import app_config so resolve_model_dir finds the repo.
    # It must go AFTER `from __future__ import annotations`: a __future__ import
    # has to be the first statement in the file, so prepending anything above it
    # is a SyntaxError and the Space would fail to boot.
    app = staging / "app.py"
    lines = app.read_text().splitlines(keepends=True)
    inject = "import app_config  # noqa: F401  (sets FINTRIAGE_MODEL_REPO)\n"
    future = next(
        (i for i, line in enumerate(lines) if line.startswith("from __future__ import")),
        None,
    )
    if future is None:
        raise SystemExit("app.py has no __future__ import; check the injection point")
    lines.insert(future + 1, inject)
    app.write_text("".join(lines))
    # Prove the result parses before it is ever uploaded.
    import ast

    ast.parse(app.read_text(), filename="app.py")

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
    ap.add_argument("--zerogpu", action="store_true",
                    help="ZeroGPU flavour (zero-a10g): the only free tier that hosts a "
                         "PRIVATE Gradio Space. cpu-basic returns HTTP 402 without PRO.")
    ap.add_argument("--bundle-weights", action="store_true",
                    help="ship the checkpoint inside the Space via LFS instead of "
                         "pulling it from a model repo. Avoids needing any token in "
                         "the Space when the model repo is private.")
    args = ap.parse_args()

    from huggingface_hub import HfApi, get_token

    # Prefer an explicit env var; otherwise use the token `hf auth login` stored.
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit(
            "Not authenticated.\n"
            "  hf auth login              (browser or token)\n"
            "  or export HF_TOKEN=hf_...  (write scope)"
        )

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

    staging = stage_space(args.model_repo, zerogpu=args.zerogpu,
                          bundle_weights=args.bundle_weights)
    hardware = "zero-a10g" if args.zerogpu else None
    # Creating an existing ZeroGPU Space again is counted against the free
    # 2-Space limit and fails with 402, so only create when it is absent.
    try:
        existing = api.space_info(args.space)
        print(f"space exists: private={existing.private} "
              f"hardware={existing.runtime.requested_hardware if existing.runtime else '?'}")
    except Exception:
        api.create_repo(args.space, repo_type="space", space_sdk="gradio",
                        space_hardware=hardware, exist_ok=True, private=private)
        print(f"created space ({hardware or 'cpu-basic'}, "
              f"{'private' if private else 'PUBLIC'})")
    api.upload_folder(folder_path=str(staging), repo_id=args.space, repo_type="space")
    info = api.space_info(args.space)
    print(f"visibility: {'private' if info.private else 'PUBLIC'}"
          f" | hardware: {info.runtime.requested_hardware if info.runtime else '?'}")
    url = f"https://huggingface.co/spaces/{args.space}"
    print(f"\nspace -> {url}")
    print("It builds for a few minutes. Verify at the URL above before calling it deployed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
