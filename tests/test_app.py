"""Application-layer tests that need no checkpoint: chart builders, HTML
fragments, the example set, and the config contract the UI depends on."""
import numpy as np
import pandas as pd
import pytest

from src import charts
from src.app_theme import CATEGORY_COLORS, CSS, DISCLAIMER
from src.config import display_name, id_to_label, label_names, label_to_id, load_config
from src.inference.examples import EXAMPLES, example_texts
from src.ui_blocks import comparison_table, empty_state, esc, hero, note, tiles

CFG = load_config()
LABELS = label_names(CFG)
DISPLAY = [display_name(label, CFG) for label in LABELS]


class TestConfigContract:
    def test_five_labels(self):
        assert len(LABELS) == 5

    def test_every_label_has_a_display_name(self):
        assert all(display_name(label, CFG) != label for label in LABELS)

    def test_display_names_are_unique(self):
        assert len(set(DISPLAY)) == 5

    def test_every_display_name_has_a_colour(self):
        assert set(DISPLAY) == set(CATEGORY_COLORS)

    def test_id_mapping_is_a_bijection(self):
        forward, backward = label_to_id(CFG), id_to_label(CFG)
        assert all(backward[i] == name for name, i in forward.items())

    def test_threshold_is_sane(self):
        assert 0.0 < CFG["inference"]["review_threshold"] < 1.0

    def test_upload_limits_are_set(self):
        assert CFG["app"]["max_upload_mb"] > 0 and CFG["app"]["max_rows"] > 0

    def test_split_fractions_sum_to_one(self):
        sp = CFG["split"]
        assert sp["train"] + sp["val"] + sp["test"] == pytest.approx(1.0)


class TestExamples:
    def test_covers_all_five_classes_plus_an_ambiguous_one(self):
        names = [name for name, _ in EXAMPLES]
        assert len(EXAMPLES) == 6
        for display in DISPLAY:
            assert display in names

    def test_examples_are_substantial(self):
        assert all(len(text) > 120 for _, text in EXAMPLES)

    def test_gradio_shape(self):
        rows = example_texts()
        assert all(isinstance(r, list) and len(r) == 1 for r in rows)

    def test_examples_are_not_dataset_rows(self):
        """Guard against a real narrative leaking into the UI. The source
        narratives are lowercased and stripped of punctuation; ours are not."""
        for _, text in EXAMPLES:
            assert any(c.isupper() for c in text)
            assert "." in text


class TestHtmlBlocks:
    def test_escaping_blocks_injection(self):
        assert "<script>" not in esc("<script>alert(1)</script>")

    def test_hero_has_the_brand_and_subtitle(self):
        html = hero()
        assert "FinTriage AI" in html
        assert "Intelligent Financial Complaint Classification" in html

    def test_tiles_render_every_item(self):
        html = tiles([("A", "1", "x"), ("B", "2", "y")])
        assert html.count("ft-tile") == 3   # wrapper + 2 tiles

    def test_note_warn_variant(self):
        assert "ft-note warn" in note("careful", warn=True)

    def test_empty_state_escapes(self):
        assert "<b>" not in empty_state("<b>hi</b>")

    def test_disclaimer_is_present_and_non_advisory(self):
        assert "reviewed before consequential decisions" in DISCLAIMER
        assert "advice" not in DISCLAIMER.lower()

    def test_css_defines_the_brand_palette(self):
        for token in ("--ft-navy", "--ft-blue", "--ft-surface"):
            assert token in CSS

    def test_css_has_a_mobile_breakpoint(self):
        assert "@media (max-width: 640px)" in CSS

    def test_comparison_table_empty_state(self):
        assert "No evaluation results" in comparison_table([], None)

    def test_comparison_table_marks_the_winner(self):
        rows = [
            {"model": "GRU", "accuracy": 0.8, "f1_macro": 0.79, "f1_weighted": 0.8,
             "train_time_sec": 50, "cpu_latency_p50_ms": 4.0, "artifact_size_mb": 11.0,
             "trainable_parameters": 1_000_000},
            {"model": "DistilBERT", "accuracy": 0.85, "f1_macro": 0.84, "f1_weighted": 0.85,
             "train_time_sec": 2400, "cpu_latency_p50_ms": 90.0, "artifact_size_mb": 268.0,
             "trainable_parameters": 67_000_000},
        ]
        html = comparison_table(rows, "DistilBERT")
        assert "ft-winner" in html and "PRODUCTION" in html
        assert "67,000,000" in html


class TestCharts:
    def test_probability_bars(self):
        probs = {label: 1 / 5 for label in LABELS}
        fig = charts.probability_bars(probs)
        assert len(fig.data) == 1 and len(fig.data[0].x) == 5

    def test_category_distribution(self):
        counts = pd.Series({"Credit Card": 4, "Debt Collection": 2})
        assert len(charts.category_distribution(counts).data) == 1

    def test_donut(self):
        counts = pd.Series({"Credit Card": 1, "Retail Banking": 3})
        assert charts.category_share_donut(counts).data[0].hole == pytest.approx(0.62)

    def test_confidence_histogram_marks_the_threshold(self):
        fig = charts.confidence_histogram(np.array([0.4, 0.9, 0.95]), 0.6)
        assert any(getattr(s, "x0", None) == 60.0 for s in fig.layout.shapes)

    def test_routing_split(self):
        fig = charts.routing_split(8, 2)
        assert list(fig.data[0].x) == [8, 2]

    def test_metric_comparison(self):
        rows = [{"model": "GRU", "f1_macro": 0.8}, {"model": "DistilBERT", "f1_macro": 0.84}]
        assert len(charts.metric_comparison(rows, "f1_macro", "Macro F1").data[0].x) == 2

    def test_latency_vs_size_is_log_scaled(self):
        rows = [{"model": "GRU", "cpu_latency_p50_ms": 4.0, "artifact_size_mb": 11.0,
                 "f1_macro": 0.8}]
        fig = charts.latency_vs_size(rows)
        assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"

    def test_confusion_heatmap_is_row_normalised(self):
        matrix = [[8, 2, 0, 0, 0], [1, 9, 0, 0, 0], [0, 0, 10, 0, 0],
                  [0, 0, 0, 10, 0], [0, 0, 0, 0, 10]]
        z = charts.confusion_heatmap(matrix, DISPLAY, "t").data[0].z
        np.testing.assert_allclose(np.array(z).sum(axis=1), np.ones(5), atol=1e-6)

    def test_training_curves_have_three_series(self):
        history = [{"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_f1_macro": 0.5},
                   {"epoch": 2, "train_loss": 0.8, "val_loss": 0.7, "val_f1_macro": 0.6}]
        assert len(charts.training_curves(history, "t").data) == 3

    def test_empty_figure_carries_the_message(self):
        fig = charts.empty_figure("nothing here")
        assert fig.layout.annotations[0].text == "nothing here"


def test_app_module_imports_without_a_model():
    """app.py must import and expose a Blocks even with no checkpoint present:
    the Space should show a clear error, not fail to boot."""
    import app

    assert app.PREDICTOR is not None or app.MODEL_ERROR
    assert app.build() is not None
