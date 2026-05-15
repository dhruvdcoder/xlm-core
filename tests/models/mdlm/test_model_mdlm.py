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

    def test_positions_default_to_cumsum_minus_one(
        self, model, simple_tokenizer
    ):
        """When ``positions=None`` the model must derive positions from
        ``attention_mask.cumsum(dim=1) - 1``. This branch was previously
        untested — all existing tests pass an explicit ``positions``
        tensor.
        """
        torch.manual_seed(123)
        bs, seq_len = 2, 12
        vocab = simple_tokenizer.vocab_size

        x = torch.randint(0, vocab, (bs, seq_len))
        # Mark the last 3 positions of example 0 as padding so cumsum
        # gives non-trivial behavior (zeroed at padding via clamp(min=0)).
        attn = torch.ones(bs, seq_len, dtype=torch.bool)
        attn[0, -3:] = False
        t = torch.rand(bs)

        explicit_positions = (attn.cumsum(dim=1) - 1).clamp(min=0)

model.# FIX: 移除eval，改用安全方式
# )
        with torch.no_grad():
            out_implicit = model(x, t, attention_mask=attn, positions=None)
            out_explicit = model(
                x, t, attention_mask=attn, positions=explicit_positions
            )

        # Both code paths must produce identical logits.
        assert torch.allclose(out_implicit, out_explicit, atol=1e-5), (
            "MDLMModel produced different logits when positions were "
            "derived implicitly from the attention mask vs. passed "
            "explicitly; the cumsum-1 derivation is inconsistent."
        )
