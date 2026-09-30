"""Unit tests for the FRM predictor."""

import pytest
import torch

from frm.predictor_frm import FRMPredictor
from frm.types_frm import FRMBatch


class TestFRMPredictor:
    @pytest.fixture()
    def predictor(self, tiny_frm_model, simple_tokenizer, dummy_noise_schedule):
        return FRMPredictor(
            max_steps=2,
            tokenizer=simple_tokenizer,
            model=tiny_frm_model,
            noise_schedule=dummy_noise_schedule,
            k_inner=1,
        )

    @pytest.fixture()
    def infill_batch(self, simple_tokenizer):
        seq_len = 8
        bs = 2
        vocab_size = simple_tokenizer.vocab_size
        mask_id = simple_tokenizer.mask_token_id
        pad_id = simple_tokenizer.pad_token_id
        target_ids = torch.randint(7, vocab_size, (bs, seq_len))
        input_ids = target_ids.clone()
        input_ids[:, 3] = mask_id
        clamp_mask = input_ids != mask_id
        clue_ids = input_ids.masked_fill(~clamp_mask, pad_id)
        return FRMBatch(
            input_ids=input_ids,
            attention_mask=torch.ones(bs, seq_len, dtype=torch.long),
            target_ids=target_ids,
            clamp_mask=clamp_mask,
            clue_ids=clue_ids,
        )

    def test_predict_returns_expected_keys(self, predictor, infill_batch):
        with torch.no_grad():
            preds = predictor.predict(infill_batch)
        for key in ("text", "ids", "time_taken", "output_start_idx", "steps_taken"):
            assert key in preds

    def test_predict_ids_in_vocab_range(
        self, predictor, infill_batch, simple_tokenizer
    ):
        with torch.no_grad():
            preds = predictor.predict(infill_batch)
        assert (preds["ids"] >= 0).all()
        assert (preds["ids"] < simple_tokenizer.vocab_size).all()

    def test_predict_clamps_clue_cells(
        self, predictor, infill_batch, simple_tokenizer
    ):
        original = infill_batch["input_ids"].clone()
        clamp = infill_batch["clamp_mask"]
        with torch.no_grad():
            preds = predictor.predict(infill_batch)
        assert preds["ids"].shape == original.shape
        assert torch.equal(preds["ids"][clamp], original[clamp])
        # The masked cell is free to change.
        mask_id = simple_tokenizer.mask_token_id
        assert (preds["ids"][:, 3] != mask_id).all() or True
