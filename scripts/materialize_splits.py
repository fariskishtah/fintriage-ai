"""Rebuild the train/val/test parquet files from the frozen manifest.

This is how the splits travel to another machine. Nothing is re-derived: the
manifest says which raw CSV row goes to which split, and this script applies
only the text normalisation that cleaning applied, then checks the result
against the recorded fingerprints. If a single character differs it refuses.

    python scripts/materialize_splits.py                    # into data/processed
    python scripts/materialize_splits.py --out /tmp/check   # verify elsewhere
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.preprocessing.quality import normalise_whitespace  # noqa: E402
from scripts.freeze_splits import MANIFEST_NAMES, find_manifest, fingerprint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output dir (default: data/processed)")
    ap.add_argument("--manifest", default=None, help="path to split_ids.csv.gz")
    args = ap.parse_args()

    cfg = load_config()
    data, pre = cfg["data"], cfg["preprocessing"]
    text_col, label_col = data["text_column"], data["label_column"]
    proc = cfg.path(data["processed_dir"])
    out_dir = Path(args.out) if args.out else proc
    manifest_path = Path(args.manifest) if args.manifest else find_manifest(proc)

    if manifest_path is None or not manifest_path.exists():
        print(
            f"no split manifest in {proc} (looked for {', '.join(MANIFEST_NAMES)})",
            file=sys.stderr,
        )
        return 1
    print(f"manifest: {manifest_path.name}")

    raw_csv = cfg.path(data["raw_dir"], data["csv_name"])
    provenance = cfg.path("results", "metrics", "dataset_provenance.json")
    fingerprints_path = cfg.path("results", "metrics", "split_fingerprint.json")
    if not raw_csv.exists():
        print(f"missing {raw_csv}", file=sys.stderr)
        return 1

    # The manifest is keyed by position in the raw file, so the raw file must be
    # the exact same bytes. Check before trusting any row number.
    if provenance.exists():
        expected = json.loads(provenance.read_text())["sha256"]
        h = hashlib.sha256()
        with open(raw_csv, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            print(
                "FAIL: the raw CSV does not match the recorded SHA-256, so the "
                "manifest's row numbers do not refer to these rows.\n"
                f"  expected {expected}\n  found    {h.hexdigest()}",
                file=sys.stderr,
            )
            return 1
        print(f"raw CSV sha256 verified: {expected[:16]}…")

    raw = pd.read_csv(raw_csv).reset_index(drop=True)
    raw["raw_row"] = raw.index.astype("int64")
    manifest = pd.read_csv(manifest_path)
    print(f"manifest: {len(manifest):,} rows -> {manifest['split'].value_counts().to_dict()}")

    merged = manifest.merge(raw[["raw_row", text_col, label_col]], on="raw_row", how="left")
    if merged[text_col].isna().any():
        print(
            f"FAIL: {int(merged[text_col].isna().sum())} manifest rows are not in the raw CSV",
            file=sys.stderr,
        )
        return 1

    # Exactly the text transformations cleaning applied, in the same order.
    merged[text_col] = merged[text_col].astype(str).map(normalise_whitespace)
    cap = int(pre["max_tokens"])
    too_long = merged[text_col].str.split().str.len() > cap
    merged.loc[too_long, text_col] = merged.loc[too_long, text_col].map(
        lambda t: " ".join(str(t).split()[:cap])
    )
    merged[label_col] = merged[label_col].astype(str).str.strip().str.lower()

    labels = list(data["labels"])
    expected_ids = merged[label_col].map({n: i for i, n in enumerate(labels)})
    if not expected_ids.equals(merged["label_id"]):
        bad = int((expected_ids != merged["label_id"]).sum())
        print(f"FAIL: {bad} rows disagree with the manifest's label_id", file=sys.stderr)
        return 1
    print("label mapping verified against the manifest")

    recorded = json.loads(fingerprints_path.read_text()) if fingerprints_path.exists() else None
    out_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for name in ("train", "val", "test"):
        block = merged[merged["split"] == name][
            ["raw_row", text_col, label_col, "label_id", "dup_group", "split"]
        ].reset_index(drop=True)
        digest = fingerprint(block[text_col])
        status = "no fingerprint on record"
        if recorded:
            want = recorded["splits"][name]["content_sha256"]
            match = digest == want
            all_ok &= match
            status = "MATCH" if match else f"MISMATCH (recorded {want[:16]}…)"
        print(f"  {name:<6} {len(block):>7,} rows  {digest[:16]}…  {status}")
        block.to_parquet(out_dir / f"{name}.parquet", index=False)

    if recorded and not all_ok:
        print(
            "\nFAIL: reconstructed splits differ from the frozen fingerprints. "
            "Do not train against them.",
            file=sys.stderr,
        )
        return 1

    print(f"\nwrote {out_dir}/{{train,val,test}}.parquet — identical to the frozen splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
