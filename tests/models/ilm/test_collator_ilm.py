"""Unit tests for ILM collators."""

import pytest
import torch

from ilm.datamodule_ilm import (
    DefaultILMCollator,
    ILMSeq2SeqCollator,
    ilm_single_segment_collate_target_fn,
    prepare_prefix_ids,
)


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

    def test_long_input_stays_within_block(
        self, simple_tokenizer, real_loglinear_schedule
    ):
        """Targets longer than block_size must not produce OOB sparse indices."""
        block_size = 16
        collator = DefaultILMCollator(
            tokenizer=simple_tokenizer,
            block_size=block_size,
            noise_schedule=real_loglinear_schedule,
            return_dense_target=False,
        )
        examples = [
            {
                "input_ids": torch.randint(
                    7, simple_tokenizer.vocab_size, (80,)
                ).tolist(),
            }
        ]
        batch = collator(examples)
        target = batch["target_ids"].coalesce()
        assert target.size(1) == block_size
        if target._nnz() > 0:
            assert int(target.indices()[1].max()) < block_size
        dense = target.to_dense()
        assert (batch["n_drops"] == dense.sum(-1)).all()


class TestPreparePrefixIdsILM:
    def test_long_prefix_cls_position_non_negative(self, simple_tokenizer):
        max_seq_len = 8
        long_prefix = list(range(7, 7 + 20))
        out = prepare_prefix_ids(
            [long_prefix],
            simple_tokenizer.pad_token_id,
            max_seq_len=max_seq_len,
            cls_token_id=simple_tokenizer.cls_token_id,
        )
        assert out["input_ids"].shape == (1, max_seq_len)
        assert int(out["cls_position"][0]) >= 0
        assert int(out["cls_position"][0]) == 0

    def test_long_prefix_without_cls_token(self, simple_tokenizer):
        """CharBPE-style tokenizers have no CLS; cls_position is start of content."""
        max_seq_len = 8
        long_prefix = list(range(7, 7 + 20))
        out = prepare_prefix_ids(
            [long_prefix],
            simple_tokenizer.pad_token_id,
            max_seq_len=max_seq_len,
            cls_token_id=None,
        )
        assert out["input_ids"].shape == (1, max_seq_len)
        assert int(out["cls_position"][0]) == 0


class TestIlmSingleSegmentCollateLongTarget:
    def test_long_target_sparse_indices_in_range(self, simple_tokenizer):
        block_size = 16
        global_offset = 16
        vocab_size = simple_tokenizer.vocab_size
        long_target = torch.randint(7, vocab_size, (80,)).tolist()
        batch = ilm_single_segment_collate_target_fn(
            [long_target],
            simple_tokenizer.pad_token_id,
            simple_tokenizer.bos_token_id,
            vocab_size,
            cls_token_id=None,
            max_seq_len=block_size,
            truncate="block",
            global_offset=global_offset,
            return_dense_target=False,
            return_dense_n_drops=True,
        )
        target = batch["target_ids"].coalesce()
        assert target.size(1) == global_offset + block_size
        if target._nnz() > 0:
            assert int(target.indices()[1].max()) < global_offset + block_size
            assert int(target.indices()[2].max()) < vocab_size
        dense = target.to_dense()
        assert (batch["n_drops"] == dense.sum(-1)).all()


class TestILMSeq2SeqCollatorLongSequences:
    def test_long_prefix_and_target(
        self, simple_tokenizer, real_loglinear_schedule
    ):
        input_block_size = 16
        block_size = 16
        collator = ILMSeq2SeqCollator(
            tokenizer=simple_tokenizer,
            noise_schedule=real_loglinear_schedule,
            block_size=block_size,
            input_block_size=input_block_size,
        )
        examples = [
            {
                "prompt_ids": torch.randint(
                    7, simple_tokenizer.vocab_size, (40,)
                ).tolist(),
                "input_ids": torch.randint(
                    7, simple_tokenizer.vocab_size, (80,)
                ).tolist(),
            }
        ]
        batch = collator(examples)
        assert batch["input_ids"].shape == (
            1,
            input_block_size + block_size,
        )
        assert int(batch["cls_position"][0]) >= 0
        target = batch["target_ids"].coalesce()
        if target._nnz() > 0:
            assert (
                int(target.indices()[1].max())
                < input_block_size + block_size
            )
        dense = target.to_dense()
        assert (batch["n_drops"] == dense.sum(-1)).all()
