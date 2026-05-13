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

    # -- ARLM-specific test beyond the base mixin --

    def test_causal_mask_blocks_future_tokens(self, model, simple_tokenizer):
        """Perturbing a *future* token must not change the logits at
        any earlier position when a strict lower-triangular causal mask
        is supplied. This verifies the model honours the (B, L, L)
        causal mask and is the core invariant the loss path relies on.
        """
        bs, seq_len = 2, 12
        vocab = simple_tokenizer.vocab_size

        torch.manual_seed(0)
        x = torch.randint(0, vocab, (bs, seq_len))
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        mask = causal.unsqueeze(0).expand(bs, -1, -1).contiguous()
        positions = torch.arange(seq_len).unsqueeze(0).expand(bs, -1)

        model.eval()
        with torch.no_grad():
            logits_a = model(x, attention_mask=mask, positions=positions)
            # Perturb the LAST token only; logits at positions [:, :-1, :]
            # must be unchanged.
            x_perturbed = x.clone()
            x_perturbed[:, -1] = (x[:, -1] + 1) % vocab
            logits_b = model(
                x_perturbed, attention_mask=mask, positions=positions
            )

        assert torch.allclose(
            logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-5
        ), (
            "ARLM logits at earlier positions changed when a future token "
            "was perturbed — the causal mask is leaking information."
        )
