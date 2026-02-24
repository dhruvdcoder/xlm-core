"""Unit tests for ARLM collators."""

import pytest
import torch

from arlm.datamodule_arlm import DefaultARLMCollator
from tests.models._base import BaseCollatorTests


class TestDefaultARLMCollator(BaseCollatorTests):
    """Tests for :class:`DefaultARLMCollator`."""

    @pytest.fixture()
    def collator(self, simple_tokenizer, dummy_noise_schedule):
        return DefaultARLMCollator(
            tokenizer=simple_tokenizer,
            block_size=32,
            noise_schedule=dummy_noise_schedule,
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

    # -- ARLM-specific tests (beyond the base mixin) --

    def test_target_has_ignore_index(self, collator, raw_examples):
        """Target should contain -100 at ignored positions (prompt or padding)."""
        batch = collator(raw_examples)
        assert (batch["target_ids"] == -100).any()
