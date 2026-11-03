"""Acquire the dataset used by the referenced project.

The Kaggle dataset endpoint for this slug serves the bundle without
authentication (verified 2026-11-03), so no credentials are required. If that
ever changes the script says exactly what to do instead of failing silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_config  # noqa: E402

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) fintriage-ai/1.0"


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as out:
        if resp.status != 200:
            raise RuntimeError(f"download failed: HTTP {resp.status}")
        shutil.copyfileobj(resp, out, length=1 << 20)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    raw_dir = cfg.path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / cfg["data"]["csv_name"]

    if csv_path.exists() and not args.force:
        print(f"already present: {csv_path}")
    else:
        archive = raw_dir / "kaggle_bundle.zip"
        print(f"downloading {cfg['data']['kaggle_download_url']}")
        try:
            _download(cfg["data"]["kaggle_download_url"], archive)
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: {exc}\n"
                "The public endpoint did not serve the bundle. Either place "
                f"{cfg['data']['csv_name']} in {raw_dir} by hand, or install the "
                "Kaggle CLI with credentials and run:\n"
                f"  kaggle datasets download -d {cfg['data']['kaggle_slug']} -p {raw_dir} --unzip",
                file=sys.stderr,
            )
            return 1
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            print("archive contents:", names)
            target = next((n for n in names if n.endswith(cfg["data"]["csv_name"])), None)
            if target is None:
                raise RuntimeError(f"{cfg['data']['csv_name']} not found in archive: {names}")
            with zf.open(target) as src, open(csv_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
        archive.unlink()

    # Provenance record: what we actually got, not what we expected to get.
    import pandas as pd

    df = pd.read_csv(csv_path)
    provenance = {
        "kaggle_slug": cfg["data"]["kaggle_slug"],
        "download_url": cfg["data"]["kaggle_download_url"],
        "file": cfg["data"]["csv_name"],
        "file_size_bytes": csv_path.stat().st_size,
        "sha256": sha256(csv_path),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "label_counts": df[cfg["data"]["label_column"]].value_counts().to_dict()
        if cfg["data"]["label_column"] in df.columns
        else None,
        "license_note": (
            "Derived from US CFPB Consumer Complaint Database, a public-domain US "
            "government dataset. Redistributed on Kaggle by shashwatwork. Narratives "
            "are already scrubbed of PII by the CFPB before publication. Raw rows are "
            "not committed to this repository."
        ),
    }
    out = cfg.path("results", "metrics", "dataset_provenance.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in provenance.items() if k != "license_note"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
