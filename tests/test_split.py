import numpy as np
import pandas as pd
import pytest

from src.preprocessing.split import (
    SplitSizes,
    assert_no_group_leakage,
    assert_no_text_leakage,
    group_stratified_split,
    split_summary,
)

LABELS = ["credit_card", "credit_reporting", "debt_collection", "mortgages_and_loans", "retail_banking"]


def make_frame(n_per_class=200, group_size=1):
    rows = []
    gid = 0
    for label in LABELS:
        for i in range(n_per_class // group_size):
            gid += 1
            for j in range(group_size):
                rows.append({"narrative": f"{label} complaint {gid} member {j}", "product": label, "dup_group": gid})
    return pd.DataFrame(rows)


def test_split_sizes_are_approximately_right():
    df = make_frame()
    df["split"] = group_stratified_split(df, "product", "dup_group", SplitSizes(0.8, 0.1, 0.1))
    shares = df["split"].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.8, abs=0.03)
    assert shares["val"] == pytest.approx(0.1, abs=0.03)
    assert shares["test"] == pytest.approx(0.1, abs=0.03)


def test_every_row_is_assigned():
    df = make_frame()
    df["split"] = group_stratified_split(df, "product", "dup_group", SplitSizes(0.8, 0.1, 0.1))
    assert df["split"].isna().sum() == 0
    assert set(df["split"]) == {"train", "val", "test"}


def test_stratification_preserves_class_shares():
    df = make_frame()
    df["split"] = group_stratified_split(df, "product", "dup_group", SplitSizes(0.7, 0.15, 0.15))
    overall = df["product"].value_counts(normalize=True)
    for name in ("train", "val", "test"):
        block = df[df["split"] == name]["product"].value_counts(normalize=True)
        for label in LABELS:
            assert block[label] == pytest.approx(overall[label], abs=0.05)


def test_groups_never_span_splits():
    df = make_frame(n_per_class=200, group_size=4)
    df["split"] = group_stratified_split(df, "product", "dup_group", SplitSizes(0.8, 0.1, 0.1))
    assert_no_group_leakage(df, "dup_group", "split")


def test_group_leakage_is_detected():
    df = make_frame(n_per_class=10, group_size=2)
    df["split"] = ["train", "test"] * (len(df) // 2)
    with pytest.raises(AssertionError, match="span more than one split"):
        assert_no_group_leakage(df, "dup_group", "split")


def test_text_leakage_is_detected():
    df = pd.DataFrame(
        {"narrative": ["same text here", "same text here"], "product": ["credit_card"] * 2,
         "split": ["train", "test"]}
    )
    with pytest.raises(AssertionError, match="identical narratives"):
        assert_no_text_leakage(df, "narrative", "split")


def test_split_is_deterministic_for_a_seed():
    df = make_frame()
    a = group_stratified_split(df, "product", "dup_group", SplitSizes(0.8, 0.1, 0.1), seed=42)
    b = group_stratified_split(df, "product", "dup_group", SplitSizes(0.8, 0.1, 0.1), seed=42)
    assert a.equals(b)


def test_summary_shape():
    df = make_frame()
    df["split"] = group_stratified_split(df, "product", "dup_group", SplitSizes(0.8, 0.1, 0.1))
    summary = split_summary(df, "product", "split")
    assert set(summary["sizes"]) == {"train", "val", "test"}
    assert sum(summary["sizes"].values()) == len(df)


def test_sizes_normalise():
    s = SplitSizes(8, 1, 1).normalised()
    assert s.train == pytest.approx(0.8)
