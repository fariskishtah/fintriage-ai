import numpy as np
import pandas as pd
import pytest

from src.preprocessing.quality import (
    assign_duplicate_groups,
    clean_dataframe,
    normalise_whitespace,
    token_count,
)
from src.preprocessing.vectorizer import WordVectorizer, tokenize

LABELS = [
    "credit_card", "credit_reporting", "debt_collection",
    "mortgages_and_loans", "retail_banking",
]


def test_normalise_whitespace_collapses_and_strips():
    assert normalise_whitespace("  a\t\tb\n c  ") == "a b c"
    assert normalise_whitespace("a\x00b") == "a b"


def test_token_count():
    assert token_count("one two three") == 3


def test_clean_drops_nulls_empties_and_invalid_labels():
    df = pd.DataFrame(
        {
            "narrative": ["a good long complaint here", None, "   ", "another valid complaint text"],
            "product": ["credit_card", "credit_card", "credit_card", "not_a_real_label"],
        }
    )
    clean, rep = clean_dataframe(df, "narrative", "product", LABELS, min_tokens=2)
    assert rep.n_null_text == 1
    assert rep.n_empty_text == 1
    assert rep.n_invalid_label == 1
    assert len(clean) == 1
    assert set(clean["product"]).issubset(set(LABELS))


def test_clean_drops_exact_duplicates():
    text = "the bank charged me an unexpected overdraft fee twice"
    df = pd.DataFrame({"narrative": [text, text, text + " again"], "product": ["retail_banking"] * 3})
    clean, rep = clean_dataframe(df, "narrative", "product", LABELS)
    assert rep.n_exact_duplicates == 1
    assert len(clean) == 2


def test_clean_truncates_rather_than_drops_long_text():
    long_text = " ".join(["word"] * 900)
    df = pd.DataFrame({"narrative": [long_text], "product": ["credit_card"]})
    clean, rep = clean_dataframe(df, "narrative", "product", LABELS, max_tokens=600)
    assert rep.n_truncated_long == 1
    assert len(clean) == 1
    assert token_count(clean.iloc[0]["narrative"]) == 600


def test_clean_raises_on_missing_columns():
    with pytest.raises(KeyError):
        clean_dataframe(pd.DataFrame({"x": ["y"]}), "narrative", "product", LABELS)


def test_label_mapping_order_is_stable():
    from src.config import id_to_label, label_to_id, load_config

    cfg = load_config()
    forward, backward = label_to_id(cfg), id_to_label(cfg)
    assert list(forward) == list(cfg["data"]["labels"])
    assert all(backward[i] == name for name, i in forward.items())
    assert len(forward) == 5


class TestNearDuplicates:
    def test_identical_texts_group_together(self):
        text = "the collector called me repeatedly about a debt i already paid"
        groups, n_multi = assign_duplicate_groups([text, text, "a completely different narrative about mortgages"])
        assert groups[0] == groups[1]
        assert groups[2] != groups[0]
        assert n_multi == 1

    def test_near_duplicate_detected(self):
        a = "the collector called me repeatedly about a debt i already paid in full last year"
        b = "the collector called me repeatedly about a debt i already paid in full last yr"
        groups, _ = assign_duplicate_groups([a, b], jaccard_threshold=0.7)
        assert groups[0] == groups[1]

    def test_unrelated_texts_stay_apart(self):
        groups, n_multi = assign_duplicate_groups(
            [
                "my mortgage escrow payment increased without any notice at all",
                "the credit bureau refuses to remove an account that is not mine",
            ]
        )
        assert groups[0] != groups[1]
        assert n_multi == 0

    def test_empty_input(self):
        groups, n_multi = assign_duplicate_groups([])
        assert len(groups) == 0 and n_multi == 0

    def test_rejects_bad_band_config(self):
        with pytest.raises(ValueError):
            assign_duplicate_groups(["a"], num_hashes=64, bands=7)


class TestVectorizer:
    def test_tokenize(self):
        assert tokenize("Don't CHARGE me $50 twice!") == ["don't", "charge", "me", "50", "twice"]

    def test_pad_and_truncate_post(self):
        vec = WordVectorizer(vocab_size=50, max_len=4).fit(["a b c d e f"])
        out = vec.transform(["a b", "a b c d e f"])
        assert out.shape == (2, 4)
        assert out[0][2] == 0 and out[0][3] == 0   # post padding
        assert (out[1] != 0).all()                 # post truncation

    def test_oov_maps_to_index_one(self):
        vec = WordVectorizer(vocab_size=10, max_len=3).fit(["known words only"])
        out = vec.transform(["totallyunseenword"])
        assert out[0][0] == 1

    def test_reserved_indices(self):
        vec = WordVectorizer(vocab_size=10, max_len=3).fit(["a b c"])
        assert vec.word_index["<PAD>"] == 0 and vec.word_index["<OOV>"] == 1

    def test_roundtrip(self, tmp_path):
        vec = WordVectorizer(vocab_size=20, max_len=5).fit(["a b c", "b c d"])
        path = tmp_path / "vec.json"
        vec.save(path)
        again = WordVectorizer.load(path)
        assert again.word_index == vec.word_index
        assert np.array_equal(again.transform(["a b"]), vec.transform(["a b"]))

    def test_vocab_cap_respected(self):
        vec = WordVectorizer(vocab_size=5, max_len=3).fit(["a b c d e f g h"])
        assert vec.effective_vocab_size <= 5
