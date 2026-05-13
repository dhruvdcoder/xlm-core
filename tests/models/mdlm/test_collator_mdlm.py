"""Unit tests for MDLM collators."""

import pytest
import torch

from mdlm.datamodule_mdlm import DefaultMDLMCollator


class TestDefaultMDLMCollator:
    """Tests for :class:`DefaultMDLMCollator`.

    ``DefaultMDLMCollator`` requires a real :class:`NoiseSchedule` (not
    :class:`DummyNoiseSchedule`) because it calls ``noise_schedule.sample_t``
    and ``noise_schedule(t)`` during collation. The shared fixture
    ``real_loglinear_schedule`` lives in :file:`tests/conftest.py`.
    """

    @pytest.fixture()
    def block_size(self):
        # Override the parametrised root-conftest fixture so each
        # collator instance only sees a single block size.
        return 32

    @pytest.fixture()
    def collator(self, simple_tokenizer, real_loglinear_schedule, block_size):
        return DefaultMDLMCollator(
            tokenizer=simple_tokenizer,
            block_size=block_size,
            noise_schedule=real_loglinear_schedule,
        )

    @pytest.fixture()
    def raw_examples(self, simple_tokenizer):
        return [
            {
                "input_ids": torch.randint(
                    7, simple_tokenizer.vocab_size, (20,)
                ).tolist(),
                "attention_mask": [1] * 20,
                "token_type_ids": [0] * 20,
            }
            for _ in range(4)
        ]

    def test_output_has_noise_fields(self, collator, raw_examples):
        batch = collator(raw_examples)
        for key in ("noise_rate", "total_noise", "t"):
            assert key in batch, f"missing {key} in MDLM batch"
            assert batch[key].shape == (len(raw_examples),)
            assert torch.isfinite(batch[key]).all()

    def test_output_shapes(self, collator, raw_examples, block_size):
        batch = collator(raw_examples)
        n = len(raw_examples)
        assert batch["input_ids"].shape == (n, block_size)
        assert batch["attention_mask"].shape == (n, block_size)
        assert batch["target_ids"].shape == (n, block_size)

    def test_target_ids_match_input_at_unmasked(
        self, collator, raw_examples, simple_tokenizer
    ):
        batch = collator(raw_examples)
        mask_id = simple_tokenizer.mask_token_id
        unmasked = batch["input_ids"] != mask_id
        # On unmasked positions, target must equal input (collator copies
        # the originals into target_ids before masking).
        assert torch.equal(
            batch["input_ids"][unmasked],
            batch["target_ids"][unmasked],
        )

    def test_t_in_expected_range(self, collator, raw_examples):
        batch = collator(raw_examples)
        # ``sample_t`` returns values in (eps, 1].
        assert (batch["t"] > 0).all()
        assert (batch["t"] <= 1.0).all()
