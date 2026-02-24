"""Unit tests for MLM collators."""

import pytest
import torch

from mlm.datamodule_mlm import DefaultMLMCollator
from tests.models._base import BaseCollatorTests


class TestDefaultMLMCollator(BaseCollatorTests):
    """Tests for :class:`DefaultMLMCollator`."""

    @pytest.fixture()
    def collator(self, simple_tokenizer, dummy_noise_schedule):
        return DefaultMLMCollator(
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
