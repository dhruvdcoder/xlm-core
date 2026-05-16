"""Unit tests for the ARLM predictor."""

import pytest
import torch

from arlm.predictor_arlm import ARLMPredictor


class TestARLMPredictor:
    """Tests for :class:`ARLMPredictor`."""

    @pytest.fixture()
    def predictor(self, tiny_arlm_model, simple_tokenizer, dummy_noise_schedule):
        return ARLMPredictor(
            max_steps=4,
            max_length=32,
            tokenizer=simple_tokenizer,
            noise_schedule=dummy_noise_schedule,
            model=tiny_arlm_model,
            sampling_method="sample_top_k",
            top=5,
        )

    @pytest.mark.slow
    def test_predict_returns_expected_keys(self, predictor, arlm_batch):
        with torch.no_grad():
            preds = predictor.predict(arlm_batch)
        assert "text" in preds
        assert "ids" in preds
        assert "time_taken" in preds

    @pytest.mark.slow
    def test_predict_ids_in_vocab_range(self, predictor, arlm_batch, simple_tokenizer):
        with torch.no_grad():
            preds = predictor.predict(arlm_batch)
        assert (preds["ids"] >= 0).all()
        assert (preds["ids"] < simple_tokenizer.vocab_size).all()
