"""Batch CSV handling. The validation logic is exercised with a stub predictor
so these tests run without a checkpoint; test_inference.py covers the real one."""
import numpy as np
import pandas as pd
import pytest

from src.config import display_name, load_config
from src.inference.batch import (
    BatchError,
    classify_frame,
    detect_text_column,
    read_csv_safely,
    to_csv_bytes,
)
from src.inference.predictor import Prediction

CFG = load_config()
LABELS = list(CFG["data"]["labels"])
LONG = (
    "the bank charged me an overdraft fee three times in a single day after "
    "reordering my transactions and would not reverse any of them"
)


class StubPredictor:
    """Deterministic fake: cycles through the classes, confidence from text length."""

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold
        self.labels = LABELS
        self.calls = 0

    def predict_batch(self, texts, batch_size: int = 32):
        out = []
        for i, text in enumerate(texts):
            self.calls += 1
            idx = i % len(LABELS)
            confidence = 0.95 if len(text) > 60 else 0.42
            probs = np.full(len(LABELS), (1 - confidence) / (len(LABELS) - 1))
            probs[idx] = confidence
            out.append(
                Prediction(
                    label=LABELS[idx],
                    display=display_name(LABELS[idx], CFG),
                    confidence=confidence,
                    probabilities={n: float(probs[j]) for j, n in enumerate(LABELS)},
                    needs_review=confidence < self.threshold,
                    threshold=self.threshold,
                )
            )
        return out


def write_csv(tmp_path, rows, name="in.csv", encoding="utf-8", **kwargs):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False, encoding=encoding, **kwargs)
    return path


class TestColumnDetection:
    @pytest.mark.parametrize("column", ["narrative", "complaint", "text", "description", "body"])
    def test_known_names(self, column):
        assert detect_text_column(pd.DataFrame({column: [LONG], "id": [1]})) == column

    def test_case_insensitive(self):
        assert detect_text_column(pd.DataFrame({"Narrative": [LONG]})) == "Narrative"

    def test_falls_back_to_longest_text_column(self):
        frame = pd.DataFrame({"ref": ["A1", "B2"], "free_form": [LONG, LONG]})
        assert detect_text_column(frame) == "free_form"

    def test_rejects_when_no_text_column(self):
        with pytest.raises(BatchError, match="No text column"):
            detect_text_column(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}))

    def test_rejects_when_text_is_too_short_to_be_a_complaint(self):
        with pytest.raises(BatchError, match="looks like complaint text"):
            detect_text_column(pd.DataFrame({"code": ["ab", "cd"]}))


class TestReading:
    def test_reads_utf8(self, tmp_path):
        path = write_csv(tmp_path, {"narrative": [LONG]})
        frame, warnings = read_csv_safely(path, max_mb=5, max_rows=100)
        assert len(frame) == 1 and warnings == []

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(BatchError, match="could not be read"):
            read_csv_safely(tmp_path / "nope.csv", max_mb=5, max_rows=100)

    def test_rejects_wrong_extension(self, tmp_path):
        path = tmp_path / "data.xlsx"
        path.write_text("a,b\n1,2\n")
        with pytest.raises(BatchError, match="Expected a .csv"):
            read_csv_safely(path, max_mb=5, max_rows=100)

    def test_rejects_oversize_file(self, tmp_path):
        path = write_csv(tmp_path, {"narrative": [LONG] * 500})
        with pytest.raises(BatchError, match="the limit is"):
            read_csv_safely(path, max_mb=0.001, max_rows=10000)

    def test_rejects_empty_file(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(BatchError, match="empty"):
            read_csv_safely(path, max_mb=5, max_rows=100)

    def test_truncates_to_row_limit_and_warns(self, tmp_path):
        path = write_csv(tmp_path, {"narrative": [LONG] * 50})
        frame, warnings = read_csv_safely(path, max_mb=5, max_rows=10)
        assert len(frame) == 10
        assert any("only the first" in w for w in warnings)

    def test_falls_back_for_non_utf8_encoding(self, tmp_path):
        path = tmp_path / "latin.csv"
        path.write_bytes(("narrative\n" + LONG + " café\n").encode("latin-1"))
        frame, warnings = read_csv_safely(path, max_mb=5, max_rows=100)
        assert len(frame) == 1
        assert any("not valid UTF-8" in w for w in warnings)

    def test_handles_bom(self, tmp_path):
        path = tmp_path / "bom.csv"
        path.write_bytes(("narrative\n" + LONG + "\n").encode("utf-8-sig"))
        frame, _ = read_csv_safely(path, max_mb=5, max_rows=100)
        assert detect_text_column(frame) == "narrative"

    def test_skips_ragged_rows_instead_of_failing(self, tmp_path):
        path = tmp_path / "ragged.csv"
        path.write_text(f"narrative\n{LONG}\n{LONG},extra,fields\n{LONG}\n")
        frame, _ = read_csv_safely(path, max_mb=5, max_rows=100)
        assert len(frame) >= 2


class TestClassifying:
    def test_happy_path(self, tmp_path):
        frame = pd.DataFrame({"narrative": [LONG, LONG + " again", "ab"]})
        result = classify_frame(frame, StubPredictor())
        assert result.n_input_rows == 3
        assert result.n_classified == 2
        assert result.n_skipped == 1
        assert set(result.frame.columns) >= {
            "predicted_category", "predicted_label", "confidence", "routing", "top_3",
        }

    def test_review_flag_follows_the_threshold(self):
        frame = pd.DataFrame({"narrative": [LONG, "ten chars."]})
        result = classify_frame(frame, StubPredictor(threshold=0.6))
        routing = result.frame["routing"].tolist()
        assert "Automated classification available" in routing
        assert result.n_review == sum(
            1 for r in routing if r == "Manual review recommended"
        )

    def test_explicit_column_overrides_detection(self):
        frame = pd.DataFrame({"narrative": ["short"], "other": [LONG]})
        result = classify_frame(frame, StubPredictor(), text_column="other")
        assert result.text_column == "other"
        assert result.n_classified == 1

    def test_unknown_explicit_column_is_rejected(self):
        with pytest.raises(BatchError, match="not in the file"):
            classify_frame(pd.DataFrame({"narrative": [LONG]}), StubPredictor(),
                           text_column="missing")

    def test_all_rows_unusable_is_rejected(self):
        frame = pd.DataFrame({"narrative": ["", " ", "a"]})
        with pytest.raises(BatchError, match="No row contained usable"):
            classify_frame(frame, StubPredictor(), text_column="narrative")

    def test_original_columns_are_preserved(self):
        frame = pd.DataFrame({"case_id": ["C1", "C2"], "narrative": [LONG, LONG + "!"]})
        result = classify_frame(frame, StubPredictor())
        assert result.frame["case_id"].tolist() == ["C1", "C2"]

    def test_top3_is_formatted(self):
        result = classify_frame(pd.DataFrame({"narrative": [LONG]}), StubPredictor())
        assert result.frame["top_3"].iloc[0].count(";") == 2

    def test_category_summary_sums_to_total(self):
        frame = pd.DataFrame({"narrative": [LONG] * 7})
        result = classify_frame(frame, StubPredictor())
        summary = result.category_summary()
        assert summary["complaints"].sum() == result.n_classified
        assert summary["share"].sum() == pytest.approx(1.0, abs=0.01)

    def test_predicted_labels_are_in_the_label_set(self):
        result = classify_frame(pd.DataFrame({"narrative": [LONG] * 5}), StubPredictor())
        assert set(result.frame["predicted_label"]).issubset(set(LABELS))


def test_csv_bytes_roundtrip():
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    blob = to_csv_bytes(frame)
    assert isinstance(blob, bytes)
    assert pd.read_csv(pd.io.common.BytesIO(blob)).equals(frame)
