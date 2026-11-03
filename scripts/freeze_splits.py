"""Freeze the splits as a portable, verifiable manifest.

Re-running the split code on another machine *should* reproduce the same
membership, but "should" is not good enough when the four models have to be
compared on identical examples. This script emits the actual row-level
assignment keyed by position in the raw CSV, and proves the manifest
reconstructs the parquet splits the local models were trained on.

    python scripts/freeze_splits.py          # write + verify
    python scripts/freeze_splits.py --check  # verify only

Output: data/processed/split_ids.csv.gz  (raw_row, label_id, split, dup_group)
        results/metrics/split_fingerprint.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, set_seed  # noqa: E402
from src.preprocessing.quality import clean_dataframe  # noqa: E402
from src.preprocessing.split import (  # noqa: E402
    SplitSizes,
    assert_no_group_leakage,
    assert_no_text_leakage,
    group_stratified_split,
    split_summary,
)

MANIFEST = "split_ids.csv.gz"


def fingerprint(texts: pd.Series) -> str:
    """Order-independent digest of a split's contents, so the check cannot be
    fooled by a reshuffle."""
    h = hashlib.sha256()
    for text in sorted(texts.astype(str)):
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def rebuild(cfg) -> pd.DataFrame:
    """Re-derive the clean frame + splits, carrying the raw CSV row position."""
    set_seed(cfg["seed"])
    data, pre, sp = cfg["data"], cfg["preprocessing"], cfg["split"]
    raw_csv = cfg.path(data["raw_dir"], data["csv_name"])
    raw = pd.read_csv(raw_csv)

    # Position in the raw file is the only id we control; the dataset's own
    # "Unnamed: 0" column is not guaranteed unique.
    raw = raw.reset_index(drop=True)
    raw["raw_row"] = raw.index.astype("int64")

    clean, _ = clean_dataframe(
        raw[["raw_row", data["text_column"], data["label_column"]]].set_index("raw_row")
        .reset_index()
        .rename(columns={})
        if False
        else raw,
        text_col=data["text_column"],
        label_col=data["label_column"],
        labels=list(data["labels"]),
        min_tokens=pre["min_tokens"],
        max_tokens=pre["max_tokens"],
        drop_exact_duplicates=pre["drop_exact_duplicates"],
        near_duplicate=pre["near_duplicate"],
    )
    # clean_dataframe narrows to [text, label]; recover raw_row by re-joining on
    # the exact cleaned text, which is unique after exact-duplicate removal.
    raw_clean_text = raw[data["text_column"]].astype(str).map(
        lambda t: " ".join(str(t).split())
    )
    lookup = (
        pd.DataFrame({"raw_row": raw["raw_row"], "_t": raw_clean_text})
        .drop_duplicates(subset="_t", keep="first")
        .set_index("_t")["raw_row"]
    )
    clean["raw_row"] = clean[data["text_column"]].map(lookup)
    missing = int(clean["raw_row"].isna().sum())
    if missing:
        # Long narratives were truncated after dedup, so their text no longer
        # matches the raw file; map those by their truncated prefix.
        trunc_lookup = {}
        cap = pre["max_tokens"]
        for row_id, text in zip(raw["raw_row"], raw_clean_text):
            key = " ".join(str(text).split()[:cap])
            trunc_lookup.setdefault(key, row_id)
        fill = clean.loc[clean["raw_row"].isna(), data["text_column"]].map(trunc_lookup)
        clean.loc[clean["raw_row"].isna(), "raw_row"] = fill
    still_missing = int(clean["raw_row"].isna().sum())
    if still_missing:
        raise RuntimeError(f"{still_missing} cleaned rows could not be traced to the raw CSV")
    clean["raw_row"] = clean["raw_row"].astype("int64")

    clean["split"] = group_stratified_split(
        clean,
        label_col=data["label_column"],
        group_col="dup_group",
        sizes=SplitSizes(sp["train"], sp["val"], sp["test"]),
        seed=cfg["seed"],
    )
    assert_no_group_leakage(clean, "dup_group", "split")
    assert_no_text_leakage(clean, data["text_column"], "split")
    clean["label_id"] = clean[data["label_column"]].map(
        {name: i for i, name in enumerate(data["labels"])}
    )
    return clean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    cfg = load_config()
    data = cfg["data"]
    text_col = data["text_column"]
    proc = cfg.path(data["processed_dir"])

    clean = rebuild(cfg)
    print(f"re-derived {len(clean):,} rows")

    # --- the verification that matters: does this match what the RNNs trained on?
    existing = {}
    for name in ("train", "val", "test"):
        path = proc / f"{name}.parquet"
        if path.exists():
            existing[name] = pd.read_parquet(path)

    report: dict = {"splits": {}, "verified_against_parquet": bool(existing)}
    ok = True
    for name in ("train", "val", "test"):
        block = clean[clean["split"] == name]
        digest = fingerprint(block[text_col])
        entry = {
            "n_rows": int(len(block)),
            "content_sha256": digest,
            "raw_row_min": int(block["raw_row"].min()),
            "raw_row_max": int(block["raw_row"].max()),
            "label_counts": {k: int(v) for k, v in block[data["label_column"]].value_counts().items()},
            "n_duplicate_groups": int(block["dup_group"].nunique()),
        }
        if name in existing:
            theirs = fingerprint(existing[name][text_col])
            entry["parquet_sha256"] = theirs
            entry["matches_parquet"] = digest == theirs
            entry["parquet_n_rows"] = int(len(existing[name]))
            if digest != theirs:
                ok = False
            print(
                f"  {name:<6} {len(block):>7,} rows  "
                f"{'MATCH' if digest == theirs else 'MISMATCH'}  {digest[:16]}"
            )
        else:
            print(f"  {name:<6} {len(block):>7,} rows  (no parquet to compare)")
        report["splits"][name] = entry

    # Cross-split leakage, stated as a number rather than assumed.
    sets = {n: set(clean[clean["split"] == n][text_col]) for n in ("train", "val", "test")}
    report["cross_split_text_overlap"] = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    groups = clean.groupby("dup_group")["split"].nunique()
    report["duplicate_groups_spanning_splits"] = int((groups > 1).sum())
    report["seed"] = cfg["seed"]
    report["dataset_sha256"] = json.loads(
        (cfg.path("results", "metrics", "dataset_provenance.json")).read_text()
    )["sha256"]

    if not ok:
        print(
            "\nFAIL: the re-derived split does not match the parquet files the local "
            "models were trained on. Do not train DistilBERT against this manifest.",
            file=sys.stderr,
        )
        return 1

    if not args.check:
        manifest = clean[["raw_row", "label_id", "split", "dup_group"]].sort_values("raw_row")
        out = proc / MANIFEST
        manifest.to_csv(out, index=False, compression="gzip")
        size_kb = out.stat().st_size / 1024
        report["manifest_file"] = MANIFEST
        report["manifest_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
        report["manifest_size_kb"] = round(size_kb, 1)
        print(f"\nwrote {out} ({size_kb:.0f} KB)")
        fp = cfg.path("results", "metrics", "split_fingerprint.json")
        fp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {fp}")
    else:
        print("\n" + json.dumps(report["cross_split_text_overlap"], indent=2))

    print(
        f"\nleakage: {report['duplicate_groups_spanning_splits']} duplicate groups span splits; "
        f"text overlap train/test = {report['cross_split_text_overlap']['train_test']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
