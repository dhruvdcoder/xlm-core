"""Unit tests for the ILM model architecture."""

import pytest
import torch

from tests.models._base import BaseModelTests


class TestRotaryTransformerILMModel(BaseModelTests):
    """Tests for :class:`RotaryTransformerILMModel`.

    The ILM model returns ``(vocab_logits, length_logits)`` where
    ``length_logits`` is ``None`` for the base model.  The ``run_forward``
    fixture unpacks the tuple so that the base mixin receives a plain
    logits tensor.
    """

    @pytest.fixture()
    def model(self, tiny_ilm_model):
        return tiny_ilm_model

    @pytest.fixture()
    def run_forward(self, model, simple_tokenizer):
        def _run(batch_size=2, seq_len=16, partial_mask=False):
            x = torch.randint(
                0, simple_tokenizer.vocab_size, (batch_size, seq_len)
            )
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            if partial_mask:
                mask[:, -4:] = False
            positions = torch.arange(seq_len).unsqueeze(0).expand(
                batch_size, -1
            )
            vocab_logits, _ = model(
                x, attention_mask=mask, positions=positions
            )
            return vocab_logits

        return _run

    # -- ILM-specific test beyond the base mixin --

    def test_length_logits_is_none(self, model, simple_tokenizer):
        """RotaryTransformerILMModel returns None for length_logits."""
        batch_size, seq_len = 2, 16
        x = torch.randint(
            0, simple_tokenizer.vocab_size, (batch_size, seq_len)
        )
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        positions = torch.arange(seq_len).unsqueeze(0).expand(
            batch_size, -1
        )
        _, length_logits = model(
            x, attention_mask=mask, positions=positions
        )
        assert length_logits is None
