"""Unit tests for the MDLM predictor."""

import pytest
import torch

from mdlm.predictor_mdlm import MDLMPredictor
from mdlm.types_mdlm import MDLMBatch


class TestMDLMPredictor:
    """Tests for :class:`MDLMPredictor`.

    ``MDLMPredictor.predict`` consumes ``noise_schedule(t)`` at every step,
    so it needs a real schedule. ``real_loglinear_schedule`` (in
    :file:`tests/conftest.py`) provides one.
    """

    @pytest.fixture()
    def predictor(self, tiny_mdlm_model, simple_tokenizer, real_loglinear_schedule):
        return MDLMPredictor(
            max_steps=2,
            tokenizer=simple_tokenizer,
            model=tiny_mdlm_model,
            noise_schedule=real_loglinear_schedule,
            top_k=5,
        )

    @pytest.fixture()
    def prefix_batch(self, simple_tokenizer):
        """A batch with one mask in the middle of a short sequence."""
        seq_len = 8
        bs = 2
        input_ids = torch.randint(7, simple_tokenizer.vocab_size, (bs, seq_len))
        input_ids[:, 3] = simple_tokenizer.mask_token_id
        attention_mask = torch.ones(bs, seq_len, dtype=torch.long)
        return MDLMBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ids=input_ids.clone(),
            noise_rate=torch.zeros(bs),
            total_noise=torch.zeros(bs),
            t=torch.ones(bs),
        )

    @pytest.mark.slow
    def test_predict_returns_expected_keys(self, predictor, prefix_batch):
        with torch.no_grad():
            preds = predictor.predict(prefix_batch)
        for key in ("text", "ids", "time_taken", "output_start_idx"):
            assert key in preds

    @pytest.mark.slow
    def test_predict_ids_in_vocab_range(
        self, predictor, prefix_batch, simple_tokenizer
    ):
        with torch.no_grad():
            preds = predictor.predict(prefix_batch)
        assert (preds["ids"] >= 0).all()
        assert (preds["ids"] < simple_tokenizer.vocab_size).all()

    @pytest.mark.slow
    def test_predict_preserves_non_mask_positions(
        self, predictor, prefix_batch, simple_tokenizer
    ):
        """Tokens that started as non-mask must be preserved verbatim."""
        original = prefix_batch["input_ids"].clone()
        mask_id = simple_tokenizer.mask_token_id
        non_mask = original != mask_id
        with torch.no_grad():
            preds = predictor.predict(prefix_batch)
        # ``predict`` may concatenate new tokens when ``max_new_tokens`` is
        # set; here it is None so the output length matches the input.
        assert preds["ids"].shape == original.shape
        assert torch.equal(preds["ids"][non_mask], original[non_mask])
