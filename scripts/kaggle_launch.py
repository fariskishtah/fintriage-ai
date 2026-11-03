"""Push and run the DistilBERT training notebook on Kaggle, headlessly.

Needs ~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token).
Nothing else: the code travels as a private Kaggle dataset, so the notebook needs
no GitHub token and no secrets.

    python scripts/kaggle_launch.py --push      # create/update dataset + kernel, start the run
    python scripts/kaggle_launch.py --status    # poll
    python scripts/kaggle_launch.py --pull      # download outputs into the project
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DATASET = "fintriage-ai-code"
KERNEL = "fintriage-distilbert-training"
COMPLAINTS = "shashwatwork/consume-complaints-dataset-fo-nlp"

# Shipped to Kaggle. Deliberately excludes data/raw, trained artifacts, the venv
# and anything resembling a credential.
INCLUDE = ["configs", "src", "scripts", "notebooks", "requirements.txt"]
MANIFEST = Path("data/processed/split_ids.csv.gz")
METRICS = ["dataset_provenance.json", "split_fingerprint.json", "split_manifest.json",
           "label_map.json", "data_quality_report.json"]


KAGGLE_BIN = str(Path(sys.executable).with_name("kaggle"))


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [KAGGLE_BIN if Path(KAGGLE_BIN).exists() else "kaggle", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        if check:
            raise SystemExit(f"kaggle {' '.join(args)} failed")
    return result


def require_credentials() -> str:
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.exists():
        raise SystemExit(
            "Missing ~/.kaggle/kaggle.json.\n"
            "  1. Open https://www.kaggle.com/settings\n"
            "  2. API -> Create New Token (downloads kaggle.json)\n"
            "  3. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ "
            "&& chmod 600 ~/.kaggle/kaggle.json"
        )
    if oct(path.stat().st_mode)[-3:] != "600":
        os.chmod(path, 0o600)
    # Read only the username; the key is never logged.
    return json.loads(path.read_text())["username"]


def stage_code(user: str) -> Path:
    staging = Path(tempfile.mkdtemp(prefix="fintriage-kaggle-")) / CODE_DATASET
    target = staging / "fintriage-ai-code"
    target.mkdir(parents=True)
    for name in INCLUDE:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, target / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"))
        elif src.exists():
            shutil.copy(src, target / name)
    (target / "data" / "processed").mkdir(parents=True)
    shutil.copy(ROOT / MANIFEST, target / MANIFEST)
    (target / "results" / "metrics").mkdir(parents=True)
    for name in METRICS:
        src = ROOT / "results" / "metrics" / name
        if src.exists():
            shutil.copy(src, target / "results" / "metrics" / name)

    # Fail loudly rather than ship a secret by accident.
    for path in target.rglob("*"):
        if path.is_file() and (
            path.name in {".env", "kaggle.json", "id_rsa"} or path.suffix in {".pem", ".key"}
        ):
            raise SystemExit(f"refusing to upload {path}")
    size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
    (staging / "dataset-metadata.json").write_text(
        json.dumps({"title": "FinTriage AI code", "id": f"{user}/{CODE_DATASET}",
                    "licenses": [{"name": "CC0-1.0"}]}, indent=2)
    )
    print(f"staged {size_mb:.1f} MB of code at {staging}")
    return staging


def stage_kernel(user: str) -> Path:
    staging = Path(tempfile.mkdtemp(prefix="fintriage-kernel-"))
    shutil.copy(ROOT / "notebooks" / "kaggle_train_distilbert.ipynb", staging / f"{KERNEL}.ipynb")
    (staging / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{user}/{KERNEL}",
                "title": "FinTriage DistilBERT Training",
                "code_file": f"{KERNEL}.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [COMPLAINTS, f"{user}/{CODE_DATASET}"],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
    )
    return staging


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--pull", action="store_true")
    args = ap.parse_args()
    user = require_credentials()
    print(f"kaggle user: {user}")

    if args.push:
        code_dir = stage_code(user)
        existing = kaggle("datasets", "list", "-m", "-s", CODE_DATASET, check=False)
        if f"{user}/{CODE_DATASET}" in existing.stdout:
            kaggle("datasets", "version", "-p", str(code_dir), "-m",
                   "sync code and frozen split manifest", "-r", "zip")
        else:
            kaggle("datasets", "create", "-p", str(code_dir), "-r", "zip")
        print("\nwaiting for the dataset to finish processing before pushing the kernel…")
        kaggle("datasets", "status", f"{user}/{CODE_DATASET}", check=False)

        kernel_dir = stage_kernel(user)
        kaggle("kernels", "push", "-p", str(kernel_dir))
        print(f"\nrunning: https://www.kaggle.com/code/{user}/{KERNEL}")

    if args.status:
        kaggle("kernels", "status", f"{user}/{KERNEL}", check=False)

    if args.pull:
        out = Path(tempfile.mkdtemp(prefix="fintriage-out-"))
        kaggle("kernels", "output", f"{user}/{KERNEL}", "-p", str(out))
        archive = out / "fintriage_artifacts.zip"
        if not archive.exists():
            found = list(out.rglob("fintriage_artifacts.zip"))
            archive = found[0] if found else None
        if archive is None:
            print(f"no fintriage_artifacts.zip in the kernel output; see {out}", file=sys.stderr)
            return 1
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(ROOT)
        print(f"extracted {archive.name} into {ROOT}")
        comparison = ROOT / "results" / "metrics" / "model_comparison.csv"
        if comparison.exists():
            print("\n" + comparison.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
