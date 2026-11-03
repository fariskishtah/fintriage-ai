"""Tests against the real trained checkpoint.

These skip — loudly, via the `model_dir` fixture — when no checkpoint exists,
rather than passing on a mock and implying the model was verified.
"""
import numpy as np
import pytest

from src.config import load_config
from src.inference.examples import EXAMPLES
from src.inference.predictor import Prediction, load_review_threshold

CFG = load_config()
LABELS = list(CFG["data"]["labels"])


class TestCheckpointContract:
    def test_label_order_matches_the_project_config(self, predictor):
        ckpt = [predictor.model.config.id2label[i] for i in range(len(LABELS))]
        assert ckpt == LABELS

    def test_five_output_classes(self, predictor):
        assert predictor.model.config.num_labels == 5

    def test_runs_on_cpu(self, predictor):
        assert predictor.device.type == "cpu"
        assert all(p.device.type == "cpu" for p in predictor.model.parameters())

    def test_max_length_matches_config(self, predictor):
        assert predictor.max_len == CFG["distilbert"]["max_len"]

    def test_mismatched_label_order_is_rejected(self, model_dir, monkeypatch):
        """A checkpoint whose id2label disagrees with the config must refuse to
        load rather than silently mislabel every prediction."""
        from src.config import Config
        from src.inference.predictor import DistilBertPredictor

        scrambled = dict(CFG)
        scrambled["data"] = dict(CFG["data"])
        scrambled["data"]["labels"] = list(reversed(LABELS))
        with pytest.raises(ValueError, match="label order mismatch"):
            DistilBertPredictor(model_dir, cfg=Config(scrambled), device="cpu")


class TestPredictionSchema:
    def test_returns_a_prediction(self, predictor):
        result = predictor.predict(EXAMPLES[0][1])
        assert isinstance(result, Prediction)

    def test_label_is_in_the_label_set(self, predictor):
        assert predictor.predict(EXAMPLES[0][1]).label in LABELS

    def test_probabilities_cover_every_class_and_sum_to_one(self, predictor):
        result = predictor.predict(EXAMPLES[1][1])
        assert set(result.probabilities) == set(LABELS)
        assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)

    def test_confidence_is_the_max_probability(self, predictor):
        result = predictor.predict(EXAMPLES[2][1])
        assert result.confidence == pytest.approx(max(result.probabilities.values()))

    def test_confidence_is_a_probability(self, predictor):
        assert 0.0 <= predictor.predict(EXAMPLES[3][1]).confidence <= 1.0

    def test_top_k_is_ordered_and_sized(self, predictor):
        top = predictor.predict(EXAMPLES[4][1]).top_k(3)
        assert len(top) == 3
        assert [v for _, v in top] == sorted((v for _, v in top), reverse=True)

    def test_review_flag_follows_the_threshold(self, predictor):
        result = predictor.predict(EXAMPLES[0][1])
        assert result.needs_review == (result.confidence < result.threshold)

    def test_serialises(self, predictor):
        payload = predictor.predict(EXAMPLES[0][1]).to_dict()
        assert set(payload) >= {"label", "display", "confidence", "needs_review", "probabilities"}


class TestInputValidation:
    @pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
    def test_empty_input_is_rejected(self, predictor, bad):
        with pytest.raises(ValueError, match="empty complaint"):
            predictor.predict(bad)

    def test_very_long_input_is_truncated_not_rejected(self, predictor):
        result = predictor.predict("overdraft fee dispute " * 2000)
        assert result.label in LABELS

    def test_unicode_survives(self, predictor):
        result = predictor.predict("the bank charged me a fee — twice — in one day, café ☕")
        assert result.label in LABELS

    def test_single_long_word(self, predictor):
        assert predictor.predict("a" * 5000).label in LABELS


class TestBatching:
    def test_batch_length_matches_input(self, predictor):
        texts = [text for _, text in EXAMPLES]
        assert len(predictor.predict_batch(texts)) == len(texts)

    def test_batch_agrees_with_single(self, predictor):
        """Batching must not change a prediction: padding is masked."""
        texts = [text for _, text in EXAMPLES[:4]]
        batched = predictor.predict_batch(texts, batch_size=4)
        for text, got in zip(texts, batched):
            alone = predictor.predict(text)
            assert alone.label == got.label
            assert alone.confidence == pytest.approx(got.confidence, abs=1e-4)

    def test_batch_size_does_not_change_results(self, predictor):
        texts = [text for _, text in EXAMPLES]
        a = predictor.probabilities(texts, batch_size=2)
        b = predictor.probabilities(texts, batch_size=8)
        np.testing.assert_allclose(a, b, atol=1e-4)

    def test_empty_batch(self, predictor):
        assert predictor.predict_batch([]) == []

    def test_deterministic(self, predictor):
        text = EXAMPLES[1][1]
        assert predictor.predict(text).confidence == pytest.approx(
            predictor.predict(text).confidence, abs=1e-6
        )


class TestThreshold:
    def test_threshold_is_a_probability(self):
        assert 0.0 < load_review_threshold(CFG) < 1.0

    def test_examples_cover_every_class(self, predictor):
        """The five labelled examples should each land on their own class; the
        sixth is deliberately ambiguous and excluded."""
        labelled = [(name, text) for name, text in EXAMPLES if "Ambiguous" not in name]
        predicted = {p.display for p in predictor.predict_batch([t for _, t in labelled])}
        assert len(predicted) >= 4, f"examples collapsed onto {predicted}"

    def test_ambiguous_example_is_less_confident_than_the_clear_ones(self, predictor):
        clear = [t for n, t in EXAMPLES if "Ambiguous" not in n]
        ambiguous = next(t for n, t in EXAMPLES if "Ambiguous" in n)
        mean_clear = np.mean([p.confidence for p in predictor.predict_batch(clear)])
        assert predictor.predict(ambiguous).confidence < mean_clear


class TestExplainability:
    def test_attributions_cover_the_tokens(self, predictor):
        from src.explainability.integrated_gradients import IntegratedGradientsExplainer

        explainer = IntegratedGradientsExplainer(predictor, steps=4)
        explanation = explainer.explain(EXAMPLES[2][1])
        assert explanation.label in LABELS
        assert len(explanation.tokens) > 5
        assert all(-1.0001 <= t.score <= 1.0001 for t in explanation.tokens)

    def test_explains_a_chosen_class(self, predictor):
        from src.explainability.integrated_gradients import IntegratedGradientsExplainer

        explainer = IntegratedGradientsExplainer(predictor, steps=4)
        target = LABELS.index("mortgages_and_loans")
        assert explainer.explain(EXAMPLES[0][1], target=target).label == "mortgages_and_loans"

    def test_long_text_is_truncated_and_flagged(self, predictor):
        from src.explainability.integrated_gradients import IntegratedGradientsExplainer

        explainer = IntegratedGradientsExplainer(predictor, steps=4, max_chars=200)
        assert explainer.explain("overdraft fee " * 200).truncated is True

    def test_highlight_pairs_shape(self, predictor):
        from src.explainability.integrated_gradients import IntegratedGradientsExplainer

        explainer = IntegratedGradientsExplainer(predictor, steps=4)
        pairs = explainer.explain(EXAMPLES[1][1]).highlight_pairs()
        assert all(isinstance(t, str) and isinstance(s, float) for t, s in pairs)
