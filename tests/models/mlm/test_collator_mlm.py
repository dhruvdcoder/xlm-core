"""Unit tests for MLM collators."""

import pytest
import torch

from mlm.datamodule_mlm import (
    DefaultInfillMLMCollator,
    DefaultMLMCollator,
    MLMInfillWithExactTargetPredCollator,
    MLMSeq2SeqPredCollator,
    finalize_fixed_positions_mask,
    prepare_prefix_ids,
    prepare_prefix_suffix_ids,
)
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


class TestMLMSeq2SeqPredCollator:
    """Tests for :class:`MLMSeq2SeqPredCollator`.

    Covers the seq2seq prediction-time path: ``input_ids`` is the
    left-padded prompt, ``target_ids`` is the right-padded suffix.
    """

    @pytest.fixture()
    def input_block_size(self):
        return 16

    @pytest.fixture()
    def block_size(self):
        # Override the parametrised root-conftest fixture to a fixed value.
        return 16

    @pytest.fixture()
    def collator(
        self,
        simple_tokenizer,
        dummy_noise_schedule,
        input_block_size,
        block_size,
    ):
        return MLMSeq2SeqPredCollator(
            tokenizer=simple_tokenizer,
            noise_schedule=dummy_noise_schedule,
            block_size=block_size,
            input_block_size=input_block_size,
            add_bos=True,
            add_eos=True,
        )

    @pytest.fixture()
    def raw_examples(self, simple_tokenizer):
        return [
            {
                "prompt_ids": [10, 11, 12],
                "input_ids": [20, 21],
            },
            {
                "prompt_ids": [13, 14],
                "input_ids": [22, 23, 24],
            },
        ]

    def test_output_shape_and_left_padded_prompt(
        self,
        collator,
        raw_examples,
        simple_tokenizer,
        input_block_size,
        block_size,
    ):
        batch = collator(raw_examples)
        n = len(raw_examples)
        assert batch["input_ids"].shape == (n, input_block_size)
        assert batch["target_ids"].shape == (n, block_size)
        assert batch["attention_mask"].shape == (n, input_block_size)
        # Prompt is left-padded: the last positions of input_ids are
        # the real prompt tokens, the first positions are pad.
        pad = simple_tokenizer.pad_token_id
        bos = simple_tokenizer.bos_token_id
        # Each example: prompt + [BOS]; the right-most prompt+BOS tokens
        # should NOT be padding.
        assert (batch["input_ids"][:, -1] == bos).all()
        # The first position is padding for shorter prompts (3+1=4 < 16).
        assert (batch["input_ids"][:, 0] == pad).all()
        assert "fixed_positions_mask" in batch
        assert batch["fixed_positions_mask"].shape == (n, input_block_size)
        assert batch["fixed_positions_mask"].all()
        hidden = ~batch["attention_mask"]
        assert batch["fixed_positions_mask"][hidden].all()

    def test_target_ids_pad_right(
        self, collator, raw_examples, simple_tokenizer, block_size
    ):
        batch = collator(raw_examples)
        pad = simple_tokenizer.pad_token_id
        eos = simple_tokenizer.eos_token_id
        # Each target should end with EOS followed by pads.
        # Example 0 has input length 2 -> target = [20, 21, EOS, pad...].
        assert batch["target_ids"][0, 0] == 20
        assert batch["target_ids"][0, 1] == 21
        assert batch["target_ids"][0, 2] == eos
        assert batch["target_ids"][0, -1] == pad

    def test_long_target_raises_by_default(
        self, collator, simple_tokenizer, block_size
    ):
        long_suffix = list(range(7, 7 + block_size + 40))
        examples = [
            {
                "prompt_ids": [10, 11, 12],
                "input_ids": long_suffix,
            }
        ]
        with pytest.raises(ValueError, match="greater than block size"):
            collator(examples)

    def test_long_target_truncated_when_flag_set(
        self,
        simple_tokenizer,
        dummy_noise_schedule,
        input_block_size,
        block_size,
    ):
        collator = MLMSeq2SeqPredCollator(
            tokenizer=simple_tokenizer,
            noise_schedule=dummy_noise_schedule,
            block_size=block_size,
            input_block_size=input_block_size,
            add_bos=True,
            add_eos=True,
            truncate_long_targets=True,
        )
        long_suffix = list(range(7, 7 + block_size + 40))
        examples = [
            {
                "prompt_ids": [10, 11, 12],
                "input_ids": long_suffix,
            }
        ]
        batch = collator(examples)
        eos = simple_tokenizer.eos_token_id
        assert batch["target_ids"].shape == (1, block_size)
        # Content is truncated to reserve the final slot for EOS.
        assert batch["target_ids"][0, -1] == eos
        assert batch["target_ids"][0, 0] == long_suffix[0]
        assert batch["target_ids"][0, -2] == long_suffix[block_size - 2]


class TestPreparePrefixIds:
    """Direct test of the ``prepare_prefix_ids`` helper.

    Used by :class:`MLMSeq2SeqPredCollator` and other seq2seq collators
    to left-pad prefixes; previously uncovered.
    """

    def test_left_pads_to_max_seq_len(self, simple_tokenizer):
        out = prepare_prefix_ids(
            [[10, 11, 12], [13, 14]],
            simple_tokenizer.pad_token_id,
            max_seq_len=8,
            truncate="block",
        )
        assert out["input_ids"].shape == (2, 8)
        pad = simple_tokenizer.pad_token_id
        assert out["input_ids"][0, -1] == 12
        assert out["input_ids"][1, -1] == 14
        assert out["input_ids"][0, 0] == pad
        assert out["input_ids"][1, 0] == pad
        # Attention mask is 1 only at non-pad positions.
        assert (
            out["attention_mask"].sum(dim=1).tolist() == [3, 2]
        )


class TestPreparePrefixSuffixIds:
    """Suffix window cap for seq2seq training (STAR and TinyGSM layouts)."""

    def test_star_hidden_tail_and_suffix_slot(self, simple_tokenizer):
        torch.manual_seed(0)
        pad = simple_tokenizer.pad_token_id
        bos = simple_tokenizer.bos_token_id
        eos = simple_tokenizer.eos_token_id
        mask = simple_tokenizer.mask_token_id
        input_block_size = 16
        block_size = 8
        max_seq_len = input_block_size + block_size

        batch = prepare_prefix_suffix_ids(
            prefix_ids=[[10, 11, 12], [13, 14]],
            suffix_ids=[[20, 21], [22, 23, 24]],
            pad_token_id=pad,
            mask_token_id=mask,
            eos_token_id=eos,
            bos_token_id=bos,
            max_seq_len=max_seq_len,
            truncate="block",
            suffix_block_size=block_size,
        )

        assert batch["input_ids"].shape == (2, max_seq_len)
        # Example 0: P=3, BOS, suffix slot 8 -> visible_len=12
        visible_len = 3 + 1 + block_size
        assert batch["attention_mask"][0, :visible_len].all()
        assert not batch["attention_mask"][0, visible_len:].any()
        assert (batch["target_ids"][0, visible_len:] == -100).all()
        # Prefix + BOS never MLM-masked
        assert batch["input_ids"][0, 0] == 10
        assert batch["input_ids"][0, 3] == bos
        assert (batch["input_ids"][0, :4] != mask).all()
        # Suffix slot layout in targets (input_ids may replace some with [MASK])
        assert batch["target_ids"][0, 4] == 20
        assert batch["target_ids"][0, 5] == 21
        assert batch["target_ids"][0, 6] == eos
        assert batch["target_ids"][0, 7] == pad
        # MLM masks only in suffix slot
        suffix_region = batch["input_ids"][0, 4:visible_len]
        assert (suffix_region == mask).any()
        fixed = batch["fixed_positions_mask"]
        assert fixed[0, :4].all()
        assert not fixed[0, 4:visible_len].any()
        assert fixed[0, visible_len:].all()
        assert finalize_fixed_positions_mask(
            fixed, batch["attention_mask"]
        ).equal(fixed)
        # Visible suffix pads take loss (target is pad, not -100).
        assert batch["target_ids"][0, 7] == pad
        assert batch["target_ids"][0, 7] != -100


class TestFixedMaskInvariant:
    """``fixed_positions_mask`` is the mutation-policy bit."""

    def test_default_mlm_hides_right_pads_when_no_loss_on_padding(
        self, simple_tokenizer, dummy_noise_schedule
    ):
        collator = DefaultMLMCollator(
            tokenizer=simple_tokenizer,
            block_size=16,
            noise_schedule=dummy_noise_schedule,
            loss_on_padding=False,
        )
        examples = [
            {
                "input_ids": [10, 11, 12],
                "attention_mask": [1, 1, 1],
                "token_type_ids": [0, 0, 0],
            }
        ]
        torch.manual_seed(0)
        batch = collator(examples)
        torch.manual_seed(0)
        again = collator(examples)
        assert torch.equal(batch["input_ids"], again["input_ids"])
        assert torch.equal(batch["target_ids"], again["target_ids"])
        assert torch.equal(batch["attention_mask"], again["attention_mask"])
        hidden = ~batch["attention_mask"]
        assert hidden.any()
        assert batch["fixed_positions_mask"][hidden].all()
        assert not batch["fixed_positions_mask"][batch["attention_mask"]].any()

    def test_visible_right_pads_are_not_fixed_when_loss_on_padding(
        self, simple_tokenizer
    ):
        torch.manual_seed(0)
        batch = prepare_prefix_suffix_ids(
            prefix_ids=[[10, 11]],
            suffix_ids=[[20]],
            pad_token_id=simple_tokenizer.pad_token_id,
            mask_token_id=simple_tokenizer.mask_token_id,
            eos_token_id=simple_tokenizer.eos_token_id,
            bos_token_id=simple_tokenizer.bos_token_id,
            max_seq_len=12,
            truncate="block",
            loss_on_padding=True,
        )
        pad = simple_tokenizer.pad_token_id
        # Last columns are right pads of the concat; they attend and take loss.
        assert batch["attention_mask"][0, -1]
        assert batch["target_ids"][0, -1] == pad
        assert not batch["fixed_positions_mask"][0, -1]
        # Prefix + BOS stay fixed.
        assert batch["fixed_positions_mask"][0, :3].all()

    def test_prepare_prefix_ids_marks_left_pads_fixed(self, simple_tokenizer):
        out = prepare_prefix_ids(
            [[10, 11, 12], [13, 14]],
            simple_tokenizer.pad_token_id,
            max_seq_len=8,
            truncate="block",
        )
        assert out["fixed_positions_mask"].all()
        hidden = ~out["attention_mask"]
        assert hidden.any()
        assert out["fixed_positions_mask"][hidden].all()

    def test_infill_clues_and_hidden_pads_are_fixed(
        self, simple_tokenizer, dummy_noise_schedule
    ):
        mask = simple_tokenizer.mask_token_id
        collator = DefaultInfillMLMCollator(
            tokenizer=simple_tokenizer,
            block_size=12,
            noise_schedule=dummy_noise_schedule,
            loss_on_padding=False,
            add_bos=False,
            add_eos=False,
        )
        examples = [
            {
                "input_ids": [10, 11, 12, 13],
                "prompt_ids": [10, mask, 12, mask],
            }
        ]
        batch = collator(examples)
        hidden = ~batch["attention_mask"]
        assert batch["fixed_positions_mask"][0, 0]
        assert not batch["fixed_positions_mask"][0, 1]
        assert batch["fixed_positions_mask"][0, 2]
        assert not batch["fixed_positions_mask"][0, 3]
        assert hidden.any()
        assert batch["fixed_positions_mask"][hidden].all()

    def test_exact_target_pred_emits_clue_fixed_mask(
        self, simple_tokenizer, dummy_noise_schedule
    ):
        mask = simple_tokenizer.mask_token_id
        collator = MLMInfillWithExactTargetPredCollator(
            tokenizer=simple_tokenizer,
            block_size=8,
            noise_schedule=dummy_noise_schedule,
            loss_on_padding=True,
            add_bos=False,
            add_eos=False,
        )
        examples = [
            {
                "prompt_ids": [10, mask, 12],
                "input_ids": [10, 11, 12],
            }
        ]
        batch = collator(examples)
        assert batch["fixed_positions_mask"][0, 0]
        assert not batch["fixed_positions_mask"][0, 1]
        assert batch["fixed_positions_mask"][0, 2]
        # Visible right pads remain editable when loss_on_padding=True.
        assert batch["attention_mask"][0, -1]
        assert not batch["fixed_positions_mask"][0, -1]
