"""Group-aware stratified train/val/test split.

Stratification keeps the class shares stable across splits; grouping keeps every
near-duplicate narrative inside one split. Both at once is not available from
scikit-learn, so we stratify over *groups* (each group labelled by its majority
class) and fill the splits group by group, largest group first, always into the
split that is furthest below its quota for that class.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitSizes:
    train: float
    val: float
    test: float

    def normalised(self) -> "SplitSizes":
        total = self.train + self.val + self.test
        return SplitSizes(self.train / total, self.val / total, self.test / total)


def group_stratified_split(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    sizes: SplitSizes,
    seed: int = 42,
) -> pd.Series:
    """Return a Series of "train"/"val"/"test" aligned to df.index."""
    sizes = sizes.normalised()
    rng = np.random.default_rng(seed)
    names = ("train", "val", "test")
    fractions = np.array([sizes.train, sizes.val, sizes.test])

    grouped = (
        df.groupby(group_col)[label_col]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("group_label")
        .reset_index()
    )
    group_size = df[group_col].value_counts().rename("size")
    grouped = grouped.join(group_size, on=group_col)

    assignment: dict[int, str] = {}
    for label, block in grouped.groupby("group_label", sort=True):
        target = block["size"].sum() * fractions
        filled = np.zeros(3, dtype=float)
        # Largest groups first so a big group cannot be forced into a full split;
        # ties broken by a seeded shuffle for reproducibility.
        order = rng.permutation(len(block))
        block = block.iloc[order].sort_values("size", ascending=False, kind="stable")
        for gid, size in zip(block[group_col], block["size"]):
            deficit = target - filled
            # Prefer whichever split is proportionally furthest behind, but never
            # starve a split whose quota rounds below one group.
            choice = int(np.argmax(deficit / np.maximum(target, 1e-9)))
            assignment[gid] = names[choice]
            filled[choice] += size
        del label

    return df[group_col].map(assignment).astype("string")


def split_summary(df: pd.DataFrame, label_col: str, split_col: str) -> dict:
    out: dict = {"sizes": {}, "label_counts": {}, "label_shares": {}}
    for name, block in df.groupby(split_col, sort=True):
        counts = block[label_col].value_counts()
        out["sizes"][str(name)] = int(len(block))
        out["label_counts"][str(name)] = {k: int(v) for k, v in counts.items()}
        out["label_shares"][str(name)] = {
            k: round(float(v) / len(block), 4) for k, v in counts.items()
        }
    return out


def assert_no_group_leakage(df: pd.DataFrame, group_col: str, split_col: str) -> None:
    spread = df.groupby(group_col)[split_col].nunique()
    bad = spread[spread > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} duplicate group(s) span more than one split: {list(bad.index[:10])}"
        )


def assert_no_text_leakage(df: pd.DataFrame, text_col: str, split_col: str) -> None:
    per_split = {name: set(block[text_col]) for name, block in df.groupby(split_col)}
    for a in ("train", "val"):
        for b in ("val", "test"):
            if a >= b:
                continue
            overlap = per_split.get(a, set()) & per_split.get(b, set())
            if overlap:
                raise AssertionError(f"{len(overlap)} identical narratives shared by {a}/{b}")
    overlap = per_split.get("train", set()) & per_split.get("test", set())
    if overlap:
        raise AssertionError(f"{len(overlap)} identical narratives shared by train/test")
