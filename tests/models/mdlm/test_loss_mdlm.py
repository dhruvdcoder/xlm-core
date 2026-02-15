"""Unit tests for the MDLM loss function."""

import pytest
import torch

from mdlm.loss_mdlm import MDLMLoss
from tests.models._base import BaseLossTests


class TestMDLMLoss(BaseLossTests):
    """Tests for :class:`MDLMLoss`."""

    @pytest.fixture()
    def loss_fn(self, tiny_mdlm_model, simple_tokenizer):
        loss = MDLMLoss(
            loss_on_padding=False,
            loss_on_visible_tokens=False,
            model=tiny_mdlm_model,
            tokenizer=simple_tokenizer,
        )
        loss.mask_token_id_tensor = torch.tensor(
            simple_tokenizer.mask_token_id, dtype=torch.long
        )
        return loss

    @pytest.fixture()
    def batch(self, mdlm_batch):
        return mdlm_batch
