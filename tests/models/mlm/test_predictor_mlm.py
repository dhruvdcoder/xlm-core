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
