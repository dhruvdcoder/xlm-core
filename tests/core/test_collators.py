"""Unit tests for base collator implementations in ``xlm.datamodule``."""

import pytest
import torch


def _make_examples(lengths, vocab_size=57):
    """Create a list of ``BaseCollatorInput``-style dicts with given lengths."""
    examples = []
    for length in lengths:
        ids = torch.randint(7, vocab_size, (length,)).tolist()
        examples.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * length,
                "token_type_ids": [0] * length,
            }
        )
    return examples


class TestDefaultCollator:
    """Tests for :class:`DefaultCollator` (no padding, stacks directly)."""

    def test_output_keys(self, default_collator):
        examples = _make_examples([32, 32])
        batch = default_collator(examples)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "token_type_ids" in batch

    def test_output_shapes(self, default_collator):
        seq_len = 32
        examples = _make_examples([seq_len, seq_len])
        batch = default_collator(examples)
        assert batch["input_ids"].shape == (2, seq_len)
        assert batch["attention_mask"].shape == (2, seq_len)
        assert batch["token_type_ids"].shape == (2, seq_len)

    def test_output_dtypes(self, default_collator):
        examples = _make_examples([32, 32])
        batch = default_collator(examples)
        assert batch["input_ids"].dtype == torch.int64
        assert batch["attention_mask"].dtype == torch.int64


class TestDefaultCollatorWithPadding:
    """Tests for :class:`DefaultCollatorWithPadding`."""

    def test_pads_short_sequences(self, padding_collator):
        examples = _make_examples([10, 20])
        batch = padding_collator(examples)
        assert batch["input_ids"].shape == (2, 32)  # block_size=32

    def test_truncates_long_sequences(self, padding_collator):
        examples = _make_examples([40, 50])
        batch = padding_collator(examples)
        assert batch["input_ids"].shape == (2, 32)

    def test_padding_uses_pad_token(self, padding_collator, simple_tokenizer):
        examples = _make_examples([10])
        batch = padding_collator(examples)
        # Positions beyond the original length should be pad tokens
        assert (
            batch["input_ids"][0, 10:] == simple_tokenizer.pad_token_id
        ).all()

    def test_attention_mask_zeros_on_padding(self, padding_collator):
        examples = _make_examples([10])
        batch = padding_collator(examples)
        assert (batch["attention_mask"][0, :10] == 1).all()
        assert (batch["attention_mask"][0, 10:] == 0).all()


class TestDefaultCollatorWithDynamicPadding:
    """Tests for :class:`DefaultCollatorWithDynamicPadding`."""

    def test_pads_to_longest_in_batch(self, dynamic_padding_collator):
        examples = _make_examples([10, 20, 15])
        batch = dynamic_padding_collator(examples)
        # Should pad to max(10, 20, 15) = 20 (< block_size=64)
        assert batch["input_ids"].shape == (3, 20)

    def test_respects_block_size_cap(self, dynamic_padding_collator):
        # block_size=64, so even if longest is 100 it should cap at 64
        examples = _make_examples([100, 50])
        batch = dynamic_padding_collator(examples)
        assert batch["input_ids"].shape[1] == 64
