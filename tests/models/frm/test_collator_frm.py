"""Unit tests for FRM collators."""

import pytest
import torch

from frm.datamodule_frm import (
    DefaultFRMCollator,
    DefaultInfillFRMCollator,
    InfillWithTargetPredFRMCollator,
)
from tests.models._base import BaseCollatorTests


class TestDefaultFRMCollator(BaseCollatorTests):
    @pytest.fixture()
    def block_size(self):
        return 32

    @pytest.fixture()
    def collator(self, simple_tokenizer, dummy_noise_schedule, block_size):
        return DefaultFRMCollator(
            tokenizer=simple_tokenizer,
            block_size=block_size,
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

    def test_clamp_mask_all_false(self, collator, raw_examples):
        batch = collator(raw_examples)
        assert "clamp_mask" in batch
        assert not batch["clamp_mask"].any()


class TestDefaultInfillFRMCollator(BaseCollatorTests):
    @pytest.fixture()
    def block_size(self):
        return 32

    @pytest.fixture()
    def collator(self, simple_tokenizer, dummy_noise_schedule, block_size):
        return DefaultInfillFRMCollator(
            tokenizer=simple_tokenizer,
            block_size=block_size,
            noise_schedule=dummy_noise_schedule,
        )

    @pytest.fixture()
    def raw_examples(self, simple_tokenizer):
        mask_id = simple_tokenizer.mask_token_id
        examples = []
        for _ in range(4):
            solution = torch.randint(
                7, simple_tokenizer.vocab_size, (20,)
            ).tolist()
            prompt = list(solution)
            prompt[5] = mask_id
            prompt[6] = mask_id
            examples.append(
                {
                    "input_ids": solution,
                    "prompt_ids": prompt,
                    "attention_mask": [1] * 20,
                    "token_type_ids": [0] * 20,
                }
            )
        return examples

    def test_clamp_mask_from_prompt(self, collator, raw_examples, simple_tokenizer):
        batch = collator(raw_examples)
        mask_id = simple_tokenizer.mask_token_id
        # After right-pad to block_size, prompt mask positions 5 and 6 are free.
        assert batch["clamp_mask"].shape == batch["input_ids"].shape
        assert not batch["clamp_mask"][:, 5].any()
        assert not batch["clamp_mask"][:, 6].any()
        assert batch["clamp_mask"][:, 0].all()
        assert (batch["clue_ids"][:, 5] == simple_tokenizer.pad_token_id).all()
        _ = mask_id


class TestInfillWithTargetPredFRMCollator:
    @pytest.fixture()
    def collator(self, simple_tokenizer, dummy_noise_schedule):
        return InfillWithTargetPredFRMCollator(
            tokenizer=simple_tokenizer,
            block_size=32,
            noise_schedule=dummy_noise_schedule,
        )

    def test_input_is_prompt_target_is_solution(
        self, collator, simple_tokenizer
    ):
        mask_id = simple_tokenizer.mask_token_id
        solution = torch.randint(7, simple_tokenizer.vocab_size, (12,)).tolist()
        prompt = list(solution)
        prompt[3] = mask_id
        batch = collator(
            [
                {
                    "input_ids": solution,
                    "prompt_ids": prompt,
                    "attention_mask": [1] * 12,
                    "token_type_ids": [0] * 12,
                }
            ]
        )
        assert batch["input_ids"][0, 3] == mask_id
        assert batch["target_ids"][0, 3] == solution[3]
        assert batch["clamp_mask"][0, 3] == False
        assert batch["clamp_mask"][0, 0] == True
