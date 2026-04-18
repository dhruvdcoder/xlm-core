"""Unit tests for the ARLM loss function."""

import pytest
import torch

from arlm.loss_arlm import ARLMLoss
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
