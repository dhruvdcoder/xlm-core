"""Unit tests for the MLM loss function."""

import pytest
import torch

from mlm.loss_mlm import MLMLoss
from tests.models._base import BaseLossTests


class TestMLMLoss(BaseLossTests):
    """Tests for :class:`MLMLoss`."""

    @pytest.fixture()
    def loss_fn(self, tiny_mlm_model, simple_tokenizer):
        loss = MLMLoss(
            loss_on_padding=False,
            loss_on_visible_tokens=False,
            model=tiny_mlm_model,
            tokenizer=simple_tokenizer,
        )
        # Simulate what Harness.configure() does
        loss.mask_token_id_tensor = torch.tensor(
            simple_tokenizer.mask_token_id, dtype=torch.long
        )
        return loss

    @pytest.fixture()
    def batch(self, mlm_batch):
        return mlm_batch
