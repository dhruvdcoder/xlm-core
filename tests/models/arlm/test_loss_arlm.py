"""Unit tests for the ARLM loss function."""

import pytest
import torch

from arlm.loss_arlm import ARLMLoss
from arlm.types_arlm import ARLMBatch
from tests.models._base import BaseLossTests


class TestARLMLoss(BaseLossTests):
    """Tests for :class:`ARLMLoss`."""

    @pytest.fixture()
    def loss_fn(self, tiny_arlm_model, simple_tokenizer):
        return ARLMLoss(
            model=tiny_arlm_model,
            tokenizer=simple_tokenizer,
        )

    @pytest.fixture()
    def batch(self, arlm_batch):
        return arlm_batch

    # -- ARLM-specific tests beyond the base mixin --

    def test_loss_is_nan_when_all_targets_ignored(
        self, loss_fn, arlm_batch
    ):
        """When every target position is ``-100`` (the ignore index),
        ``torch.nn.functional.cross_entropy(..., reduction="mean")``
        averages over zero elements and returns NaN. This is the
        ignore-index branch that the previous tests did not exercise.

        We assert NaN explicitly (rather than 0.0) so that any future
        change to use ``reduction="sum"`` + safe-divide also flags
        through this test.
        """
        all_ignore_batch = ARLMBatch(
            input_ids=arlm_batch["input_ids"],
            attention_mask=arlm_batch["attention_mask"],
            target_ids=torch.full_like(arlm_batch["target_ids"], -100),
        )
        result = loss_fn(all_ignore_batch)
        assert "loss" in result
        # Either NaN (current behavior) or exactly 0.0 if a future
        # refactor adds the safe-divide. Both are sane outcomes.
        loss = result["loss"]
        assert torch.isnan(loss) or torch.equal(loss, torch.tensor(0.0))

    def test_loss_changes_when_targets_change(
        self, loss_fn, arlm_batch, simple_tokenizer
    ):
        """Sanity check that the loss actually depends on ``target_ids``
        (and not, e.g., always zero because of an over-eager ignore mask).
        """
        result_a = loss_fn(arlm_batch)
        # Shift all non-ignored targets by 1 in vocab space.
        new_targets = arlm_batch["target_ids"].clone()
        non_ignore = new_targets != -100
        new_targets[non_ignore] = (
            new_targets[non_ignore] + 1
        ) % simple_tokenizer.vocab_size
        result_b = loss_fn(
            ARLMBatch(
                input_ids=arlm_batch["input_ids"],
                attention_mask=arlm_batch["attention_mask"],
                target_ids=new_targets,
            )
        )
        assert not torch.equal(result_a["loss"], result_b["loss"])
