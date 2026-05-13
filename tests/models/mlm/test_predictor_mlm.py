"""Unit tests for the MLM predictor."""

import pytest
import torch

from mlm.predictor_mlm import MLMPredictor


class TestMLMPredictor:
    """Tests for :class:`MLMPredictor`."""

    @pytest.fixture()
    def predictor(self, tiny_mlm_model, simple_tokenizer, dummy_noise_schedule):
        return MLMPredictor(
            max_steps=2,
            tokenizer=simple_tokenizer,
            model=tiny_mlm_model,
            noise_schedule=dummy_noise_schedule,
            top_k=5,
        )

    @pytest.mark.slow
    def test_predict_returns_expected_keys(self, predictor, mlm_batch):
        """Run a short prediction and check output dict structure."""
        with torch.no_grad():
            preds = predictor.predict(mlm_batch)
        assert "text" in preds
        assert "ids" in preds
        assert "time_taken" in preds

    @pytest.mark.slow
    def test_predict_ids_in_vocab_range(self, predictor, mlm_batch, simple_tokenizer):
        with torch.no_grad():
            preds = predictor.predict(mlm_batch)
        assert (preds["ids"] >= 0).all()
        assert (preds["ids"] < simple_tokenizer.vocab_size).all()


class TestMLMPredictorConfidenceSampling:
    """Cover the confidence-based position-selection branches.

    ``predict_single_step`` has three confidence modes — ``prob_diff``,
    ``top_prob``, and ``entropy`` (NotImplementedError). The default
    ``predictor`` fixture above only exercises the uniform-random branch
    (``confidence=None``), so these tests cover the remaining executable
    branches.
    """

    @pytest.fixture()
    def model(self, tiny_mlm_model):
        return tiny_mlm_model

    @pytest.fixture()
    def conf_batch(self, mlm_batch):
        return mlm_batch

    @pytest.mark.slow
    @pytest.mark.parametrize("confidence", ["prob_diff", "top_prob"])
    def test_predict_with_confidence(
        self, model, simple_tokenizer, dummy_noise_schedule, conf_batch, confidence
    ):
        predictor = MLMPredictor(
            max_steps=2,
            tokenizer=simple_tokenizer,
            model=model,
            noise_schedule=dummy_noise_schedule,
            top_k=5,
            confidence=confidence,
            threshold=0.5,
        )
        with torch.no_grad():
            preds = predictor.predict(conf_batch)
        assert "ids" in preds and "text" in preds
        assert (preds["ids"] >= 0).all()
        assert (preds["ids"] < simple_tokenizer.vocab_size).all()

    def test_confidence_without_threshold_raises(
        self, model, simple_tokenizer, dummy_noise_schedule
    ):
        with pytest.raises(ValueError, match="hreshold"):
            MLMPredictor(
                max_steps=2,
                tokenizer=simple_tokenizer,
                model=model,
                noise_schedule=dummy_noise_schedule,
                top_k=5,
                confidence="prob_diff",
                threshold=None,
            )

    def test_both_top_k_and_top_p_raises(
        self, model, simple_tokenizer, dummy_noise_schedule
    ):
        with pytest.raises(ValueError, match="Both top_k and top_p"):
            MLMPredictor(
                max_steps=2,
                tokenizer=simple_tokenizer,
                model=model,
                noise_schedule=dummy_noise_schedule,
                top_k=5,
                top_p=0.9,
            )
