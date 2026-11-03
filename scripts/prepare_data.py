"""Clean the raw CSV, build the splits, and write every artifact the rest of
the pipeline reads. Idempotent: re-running reproduces identical files."""
from __future__ import annotations

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


def main() -> int:
    cfg = load_config()
    set_seed(cfg["seed"])
    data, pre, sp = cfg["data"], cfg["preprocessing"], cfg["split"]

    raw_csv = cfg.path(data["raw_dir"], data["csv_name"])
    if not raw_csv.exists():
        print(f"missing {raw_csv}; run `python -m src.data.download` first", file=sys.stderr)
        return 1

    df = pd.read_csv(raw_csv)
    print(f"loaded {len(df):,} rows, columns={list(df.columns)}")

    clean, report = clean_dataframe(
        df,
        text_col=data["text_column"],
        label_col=data["label_column"],
        labels=list(data["labels"]),
        min_tokens=pre["min_tokens"],
        max_tokens=pre["max_tokens"],
        drop_exact_duplicates=pre["drop_exact_duplicates"],
        near_duplicate=pre["near_duplicate"],
    )
    print(f"clean: {report.n_input:,} -> {report.n_output:,} rows")

    clean["split"] = group_stratified_split(
        clean,
        label_col=data["label_column"],
        group_col="dup_group",
        sizes=SplitSizes(sp["train"], sp["val"], sp["test"]),
        seed=cfg["seed"],
    )
    assert_no_group_leakage(clean, "dup_group", "split")
    assert_no_text_leakage(clean, data["text_column"], "split")

    summary = split_summary(clean, data["label_column"], "split")
    print(json.dumps(summary["sizes"], indent=2))

    proc = cfg.path(data["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    label_map = {name: i for i, name in enumerate(data["labels"])}
    clean["label_id"] = clean[data["label_column"]].map(label_map)

    for name in ("train", "val", "test"):
        block = clean[clean["split"] == name]
        block.to_parquet(proc / f"{name}.parquet", index=False)
        print(f"wrote {proc / f'{name}.parquet'} ({len(block):,} rows)")

    metrics = cfg.path("results", "metrics")
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "data_quality_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    (metrics / "split_manifest.json").write_text(
        json.dumps(
            {
                "seed": cfg["seed"],
                "protocol": {"train": sp["train"], "val": sp["val"], "test": sp["test"]},
                "stratified_by": data["label_column"],
                "grouped_by": "near-duplicate group (MinHash/LSH)",
                "label_map": label_map,
                **summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (metrics / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    print(f"wrote reports to {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
