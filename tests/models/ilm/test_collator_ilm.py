"""Unit tests for ILM collators."""

import pytest
import torch

from ilm.datamodule_ilm import DefaultILMCollator


class TestDefaultILMCollator:
    """Tests for :class:`DefaultILMCollator`.

    ``DefaultILMCollator`` builds sparse ``target_ids``/``n_drops`` tensors
    from the token-drop noising pipeline. Although the constructor stores a
    ``NoiseSchedule`` and ``ilm_drop_fn`` uses one indirectly via
    ``sample_n_drops_fn``, the collator's default ``_n_drop_uniformly``
    helper does not call ``noise_schedule`` directly. We still wire a real
    schedule (the shared ``real_loglinear_schedule`` fixture) to confirm
    that construction with a real schedule does not raise.
    """

    @pytest.fixture()
    def block_size(self):
        return 32

    @pytest.fixture()
    def collator(self, simple_tokenizer, real_loglinear_schedule, block_size):
        return DefaultILMCollator(
            tokenizer=simple_tokenizer,
            block_size=block_size,
            noise_schedule=real_loglinear_schedule,
            return_dense_target=True,
        )

    @pytest.fixture()
    def raw_examples(self, simple_tokenizer):
        return [
            {
                "input_ids": torch.randint(
                    7, simple_tokenizer.vocab_size, (20,)
                ).tolist(),
            }
            for _ in range(4)
        ]

    def test_construction_with_real_schedule(self, collator):
        assert collator.noise_schedule is not None
        assert collator.block_size == 32

    def test_loss_on_padding_true_rejected(
        self, simple_tokenizer, real_loglinear_schedule
    ):
        with pytest.raises(AssertionError):
            DefaultILMCollator(
                tokenizer=simple_tokenizer,
                block_size=32,
                noise_schedule=real_loglinear_schedule,
                loss_on_padding=True,
            )

    def test_output_has_expected_keys(self, collator, raw_examples):
        batch = collator(raw_examples)
        for key in (
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "target_ids",
            "n_drops",
            "cls_position",
        ):
            assert key in batch, f"missing {key} in ILM batch"

    def test_output_shapes(
        self, collator, raw_examples, block_size, simple_tokenizer
    ):
        batch = collator(raw_examples)
        n = len(raw_examples)
        assert batch["input_ids"].shape == (n, block_size)
        assert batch["attention_mask"].shape == (n, block_size)
        assert batch["token_type_ids"].shape == (n, block_size)
        # target_ids is dense thanks to return_dense_target=True.
        assert batch["target_ids"].shape == (
            n,
            block_size,
            simple_tokenizer.vocab_size,
        )
        assert batch["n_drops"].shape == (n, block_size)
        assert batch["cls_position"].shape == (n,)

    def test_cls_position_defaults_to_zero(self, collator, raw_examples):
        batch = collator(raw_examples)
        assert (batch["cls_position"] == 0).all()
