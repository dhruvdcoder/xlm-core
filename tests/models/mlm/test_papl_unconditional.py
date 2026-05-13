"""Unit tests for :mod:`mlm.papl_unconditional`."""

import pytest
import torch

from mlm.papl_unconditional import (
    PAPL_DEFAULT_LENGTHS,
    PaplUnconditionalCollator,
    PaplUnconditionalMLMDataset,
)


class TestPaplUnconditionalMLMDataset:
    def test_emits_examples_per_node_per_length_in_order(
        self, simple_tokenizer
    ):
        ds = PaplUnconditionalMLMDataset(
            tokenizer=simple_tokenizer,
            examples_per_node=2,
            lengths=(3, 5),
        )
        items = list(ds)
        assert len(items) == 4
        # Length pattern: [3, 3, 5, 5] preserving the order of ``lengths``.
        assert [len(x["input_ids"]) for x in items] == [3, 3, 5, 5]
        # Every entry is filled with the mask-token id.
        mask_id = simple_tokenizer.mask_token_id
        for x in items:
            assert all(t == mask_id for t in x["input_ids"])

    def test_default_lengths(self, simple_tokenizer):
        ds = PaplUnconditionalMLMDataset(
            tokenizer=simple_tokenizer, examples_per_node=1
        )
        assert ds.lengths == PAPL_DEFAULT_LENGTHS


class TestPaplUnconditionalCollator:
    def test_stacks_fixed_length_examples(self):
        # PaplUnconditionalCollator has no __init__: the tokenizer is supplied
        # by the dataset, not the collator.
        collator = PaplUnconditionalCollator()
        examples = [{"input_ids": [2, 2, 2]}, {"input_ids": [2, 2, 2]}]
        batch = collator(examples)
        assert batch["input_ids"].shape == (2, 3)
        assert torch.equal(
            batch["attention_mask"], torch.ones(2, 3, dtype=torch.long)
        )
        assert torch.equal(batch["target_ids"], batch["input_ids"])

    def test_mixed_lengths_raise(self):
        collator = PaplUnconditionalCollator()
        examples = [{"input_ids": [2, 2, 2]}, {"input_ids": [2, 2]}]
        with pytest.raises(ValueError, match="fixed length"):
            collator(examples)

    def test_target_ids_independent_of_input_ids(self):
        collator = PaplUnconditionalCollator()
        examples = [{"input_ids": [2, 2]}, {"input_ids": [2, 2]}]
        batch = collator(examples)
        batch["target_ids"][0, 0] = 17
        assert batch["input_ids"][0, 0] == 2
