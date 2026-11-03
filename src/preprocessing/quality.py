"""Data-quality audit and cleaning, including near-duplicate grouping.

Near-duplicates matter here because the source file contains many complaints
that differ only in a word or two. If they land on both sides of a split the
test score measures memorisation, not generalisation. We group them with
MinHash + LSH banding over character 5-gram shingles, then keep every member of
a group inside one split.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np
import pandas as pd

_WS = re.compile(r"\s+")
_NON_PRINT = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


def normalise_whitespace(text: str) -> str:
    return _WS.sub(" ", _NON_PRINT.sub(" ", str(text))).strip()


def token_count(text: str) -> int:
    return len(str(text).split())


@dataclass
class QualityReport:
    n_input: int = 0
    n_null_text: int = 0
    n_null_label: int = 0
    n_empty_text: int = 0
    n_invalid_label: int = 0
    n_exact_duplicates: int = 0
    n_too_short: int = 0
    n_truncated_long: int = 0
    n_non_ascii_repaired: int = 0
    n_output: int = 0
    near_duplicate_groups: int = 0
    rows_in_multi_member_groups: int = 0
    length_percentiles: dict = field(default_factory=dict)
    label_counts: dict = field(default_factory=dict)
    label_shares: dict = field(default_factory=dict)
    invalid_label_values: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _shingles(text: str, k: int) -> set[str]:
    text = text.replace(" ", "")
    if len(text) <= k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _minhash_signature(shingles: set[str], num_hashes: int) -> np.ndarray:
    """Deterministic MinHash: hash each shingle once, then derive `num_hashes`
    permutations from that 64-bit value. No numpy RNG, so it is stable across
    runs and machines."""
    if not shingles:
        return np.full(num_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
    base = np.fromiter(
        (int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
         for s in shingles),
        dtype=np.uint64,
        count=len(shingles),
    )
    # a_i, b_i are fixed odd constants; (a*x + b) mod 2^64 is a cheap permutation.
    i = np.arange(1, num_hashes + 1, dtype=np.uint64)
    a = (i * np.uint64(0x9E3779B97F4A7C15)) | np.uint64(1)
    b = i * np.uint64(0xBF58476D1CE4E5B9)
    with np.errstate(over="ignore"):
        perms = (base[None, :] * a[:, None] + b[:, None])
    return perms.min(axis=1)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_duplicate_groups(
    texts: Iterable[str],
    shingle_size: int = 5,
    num_hashes: int = 64,
    bands: int = 16,
    jaccard_threshold: float = 0.8,
) -> tuple[np.ndarray, int]:
    """Return a group id per row plus the number of multi-member groups.

    Candidate pairs come from LSH banding; each candidate pair is then verified
    with the real MinHash Jaccard estimate before being merged, so the banding
    recall knob cannot silently over-merge.
    """
    texts = list(texts)
    n = len(texts)
    if n == 0:
        return np.empty(0, dtype=np.int64), 0
    if num_hashes % bands != 0:
        raise ValueError("num_hashes must be divisible by bands")
    rows = num_hashes // bands

    sigs = np.empty((n, num_hashes), dtype=np.uint64)
    for i, text in enumerate(texts):
        sigs[i] = _minhash_signature(_shingles(text, shingle_size), num_hashes)

    uf = _UnionFind(n)
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        chunk = sigs[:, band * rows : (band + 1) * rows]
        for i in range(n):
            key = chunk[i].tobytes()
            buckets.setdefault(key, []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            # Cap pathological buckets: compare each member to the bucket anchor.
            anchor = members[0]
            for other in members[1:]:
                est = float((sigs[anchor] == sigs[other]).mean())
                if est >= jaccard_threshold:
                    uf.union(anchor, other)

    roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64, count=n)
    _, groups, counts = np.unique(roots, return_inverse=True, return_counts=True)
    n_multi = int((counts > 1).sum())
    return groups.astype(np.int64), n_multi


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    labels: list[str],
    min_tokens: int = 3,
    max_tokens: int = 600,
    drop_exact_duplicates: bool = True,
    near_duplicate: dict | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    rep = QualityReport(n_input=int(len(df)))
    df = df.copy()

    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(
            f"expected columns {text_col!r} and {label_col!r}; got {list(df.columns)}"
        )
    df = df[[text_col, label_col]]

    rep.n_null_text = int(df[text_col].isna().sum())
    rep.n_null_label = int(df[label_col].isna().sum())
    df = df.dropna(subset=[text_col, label_col])

    raw = df[text_col].astype(str)
    df[text_col] = raw.map(normalise_whitespace)
    rep.n_non_ascii_repaired = int((df[text_col] != raw).sum())

    rep.n_empty_text = int((df[text_col].str.len() == 0).sum())
    df = df[df[text_col].str.len() > 0]

    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    valid = df[label_col].isin(labels)
    rep.n_invalid_label = int((~valid).sum())
    rep.invalid_label_values = (
        df.loc[~valid, label_col].value_counts().head(20).to_dict()
    )
    df = df[valid]

    if drop_exact_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=[text_col])
        rep.n_exact_duplicates = int(before - len(df))

    lengths = df[text_col].map(token_count)
    rep.length_percentiles = {
        f"p{p}": float(np.percentile(lengths, p)) for p in (1, 5, 25, 50, 75, 95, 99)
    }
    rep.length_percentiles["mean"] = float(lengths.mean())
    rep.length_percentiles["max"] = int(lengths.max())

    short = lengths < min_tokens
    rep.n_too_short = int(short.sum())
    df, lengths = df[~short], lengths[~short]

    long_mask = lengths > max_tokens
    rep.n_truncated_long = int(long_mask.sum())
    if long_mask.any():
        # Truncate rather than drop: a long complaint is still a valid complaint.
        df.loc[long_mask, text_col] = df.loc[long_mask, text_col].map(
            lambda t: " ".join(str(t).split()[:max_tokens])
        )

    df = df.reset_index(drop=True)

    nd = near_duplicate or {}
    if nd.get("enabled", False):
        groups, n_multi = assign_duplicate_groups(
            df[text_col],
            shingle_size=nd.get("shingle_size", 5),
            num_hashes=nd.get("num_hashes", 64),
            bands=nd.get("bands", 16),
            jaccard_threshold=nd.get("jaccard_threshold", 0.8),
        )
        df["dup_group"] = groups
        sizes = df["dup_group"].map(df["dup_group"].value_counts())
        rep.near_duplicate_groups = n_multi
        rep.rows_in_multi_member_groups = int((sizes > 1).sum())
    else:
        df["dup_group"] = np.arange(len(df), dtype=np.int64)

    rep.n_output = int(len(df))
    counts = df[label_col].value_counts()
    rep.label_counts = {k: int(v) for k, v in counts.items()}
    rep.label_shares = {k: round(float(v) / len(df), 4) for k, v in counts.items()}
    return df, rep
