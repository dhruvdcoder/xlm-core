"""Unit tests for the ARLM model architecture."""

import pytest
import torch

from tests.models._base import BaseModelTests


class TestRotaryTransformerARLMModel(BaseModelTests):
    """Tests for :class:`RotaryTransformerARLMModel`."""

    @pytest.fixture()
    def model(self, tiny_arlm_model):
        return tiny_arlm_model

    @pytest.fixture()
    def run_forward(self, model, simple_tokenizer):
        def _run(batch_size=2, seq_len=16, partial_mask=False):
            x = torch.randint(
                0, simple_tokenizer.vocab_size, (batch_size, seq_len)
            )
            causal = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool)
            )
            mask = causal.unsqueeze(0).expand(batch_size, -1, -1).clone()
            if partial_mask:
                mask[:, :, -4:] = False
            positions = torch.arange(seq_len).unsqueeze(0).expand(
                batch_size, -1
            )
            return model(x, attention_mask=mask, positions=positions)

        return _run
