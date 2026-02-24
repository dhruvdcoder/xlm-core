"""Unit tests for the MDLM model architecture."""

import pytest
import torch

from tests.models._base import BaseModelTests


class TestMDLMModel(BaseModelTests):
    """Tests for :class:`MDLMModel`."""

    @pytest.fixture()
    def model(self, tiny_mdlm_model):
        return tiny_mdlm_model

    @pytest.fixture()
    def run_forward(self, model, simple_tokenizer):
        def _run(batch_size=2, seq_len=16, partial_mask=False):
            x = torch.randint(
                0, simple_tokenizer.vocab_size, (batch_size, seq_len)
            )
            t = torch.rand(batch_size)
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            if partial_mask:
                mask[:, -4:] = False
            positions = torch.arange(seq_len).unsqueeze(0).expand(
                batch_size, -1
            )
            return model(x, t, attention_mask=mask, positions=positions)

        return _run
