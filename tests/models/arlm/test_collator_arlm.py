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

    def test_target_padded_to_input_length(self, collator, raw_examples):
        """Short examples must pad target_ids to the same length as input_ids."""
        batch = collator(raw_examples)
        assert batch["target_ids"].shape == batch["input_ids"].shape
        assert batch["target_ids"].shape[1] == collator.block_size


class TestDefaultARLMCollatorPrompt:
    """``DefaultARLMCollator`` with ``prompt_ids`` (single-sequence seq2seq)."""

    @pytest.fixture()
    def raw_examples(self):
        return [
            {"prompt_ids": [10, 11, 12], "input_ids": [20, 21]},
            {"prompt_ids": [10, 11], "input_ids": [20, 21, 22]},
        ]

    @pytest.fixture()
    def collator_bos(self, simple_tokenizer, dummy_noise_schedule):
        return DefaultARLMCollator(
            tokenizer=simple_tokenizer,
            block_size=32,
            noise_schedule=dummy_noise_schedule,
            add_bos=True,
            add_eos=True,
            truncate="max",
        )

    @pytest.fixture()
    def collator_no_bos(self, simple_tokenizer, dummy_noise_schedule):
        return DefaultARLMCollator(
            tokenizer=simple_tokenizer,
            block_size=32,
            noise_schedule=dummy_noise_schedule,
            add_bos=False,
            add_eos=True,
            truncate="max",
        )

    def test_batch_stacks_at_batch_max(self, collator_bos, raw_examples):
        batch = collator_bos(raw_examples)
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].shape[1] <= collator_bos.block_size
        assert batch["target_ids"].shape == batch["input_ids"].shape
        assert batch["attention_mask"].shape == batch["input_ids"].shape

    def test_bos_at_prompt_join(self, collator_bos, raw_examples, simple_tokenizer):
        batch = collator_bos(raw_examples)
        prompt = raw_examples[0]["prompt_ids"]
        answer = raw_examples[0]["input_ids"]
        p = len(prompt)
        assert batch["input_ids"][0, :p].tolist() == prompt
        assert batch["input_ids"][0, p].item() == simple_tokenizer.bos_token_id
        assert batch["input_ids"][0, p + 1 : p + 1 + len(answer)].tolist() == answer
        assert (
            batch["input_ids"][0, p + 1 + len(answer)].item()
            == simple_tokenizer.eos_token_id
        )

    def test_prompt_masked_from_loss_with_bos(
        self, collator_bos, raw_examples, simple_tokenizer
    ):
        batch = collator_bos(raw_examples)
        prompt = raw_examples[0]["prompt_ids"]
        answer = raw_examples[0]["input_ids"]
        n_ignore = len(prompt) + 1 - 1
        assert batch["target_ids"][0, :n_ignore].tolist() == [-100] * n_ignore
        first_kept = batch["target_ids"][0, n_ignore].item()
        assert first_kept == answer[0]
        assert simple_tokenizer.eos_token_id in batch["target_ids"][0].tolist()

    def test_prompt_masked_from_loss_without_bos(
        self, collator_no_bos, raw_examples, simple_tokenizer
    ):
        batch = collator_no_bos(raw_examples)
        prompt = raw_examples[0]["prompt_ids"]
        answer = raw_examples[0]["input_ids"]
        p = len(prompt)
        assert batch["input_ids"][0, :p].tolist() == prompt
        assert batch["input_ids"][0, p].item() != simple_tokenizer.bos_token_id
        assert batch["input_ids"][0, p : p + len(answer)].tolist() == answer
        n_ignore = p - 1
        assert batch["target_ids"][0, :n_ignore].tolist() == [-100] * n_ignore
        assert batch["target_ids"][0, n_ignore].item() == answer[0]
        assert simple_tokenizer.eos_token_id in batch["target_ids"][0].tolist()

    def test_pads_are_ignored(
        self, collator_bos, simple_tokenizer, dummy_noise_schedule
    ):
        examples = [
            {"prompt_ids": [10], "input_ids": [20]},
            {"prompt_ids": [10, 11, 12], "input_ids": [20, 21, 22]},
        ]
        batch = collator_bos(examples)
        row0 = batch["target_ids"][0]
        mask0 = batch["attention_mask"][0]
        pad_positions = (mask0 == 0).nonzero(as_tuple=True)[0]
        if len(pad_positions):
            # targets that predict a pad (or sit on the last pad slot) are -100
            assert (row0[pad_positions] == -100).all()
            pred_pad = pad_positions[pad_positions > 0] - 1
            assert (row0[pred_pad] == -100).all()
