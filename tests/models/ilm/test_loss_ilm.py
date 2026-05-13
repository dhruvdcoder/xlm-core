"""Unit tests for the ILM loss function."""

import pytest
import torch

from ilm.datamodule_ilm import DefaultILMCollator
from ilm.loss_ilm import ILMLossWithMaskedCE


class TestILMLossWithMaskedCEConstruction:
    """Constructor / configuration guards for :class:`ILMLossWithMaskedCE`."""

    def test_stopping_class_weight_requires_binary_ce(self):
        with pytest.raises(ValueError):
            ILMLossWithMaskedCE(
                length_loss="ce",
                stopping_class_weight=0.5,
            )

    def test_loss_on_padding_raises(self):
        with pytest.raises(ValueError, match="loss_on_padding"):
            ILMLossWithMaskedCE(loss_on_padding=True, length_loss="ce")

    def test_use_constraint_raises(self):
        with pytest.raises(NotImplementedError):
            ILMLossWithMaskedCE(length_loss="ce", use_constraint=True)


class _LengthHeadWrapper(torch.nn.Module):
    """Minimal ILM-shaped model: wraps a real ILM backbone and adds a
    tiny length head so the loss path that asserts ``length_logits is
    not None`` can be exercised in a unit test.

    This is deliberately lightweight — we just need a model object that
    returns ``(vocab_logits, length_logits)`` from a single
    ``forward(x, attention_mask, positions, cls_position)`` call.
    """

    def __init__(self, base, num_classes: int):
        super().__init__()
        self.base = base
        d_model = base.d_model
        self.length_head = torch.nn.Linear(d_model, num_classes)

    def forward(
        self,
        x_t,
        attention_mask=None,
        positions=None,
        token_type_ids=None,
        cls_position=None,
    ):
        vocab_logits, _ = self.base(
            x_t,
            attention_mask=attention_mask,
            positions=positions,
            token_type_ids=token_type_ids,
            cls_position=cls_position,
        )
        # Pool a single representation per example for the length head.
        if cls_position is not None:
            idx = cls_position.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, vocab_logits.shape[-1]
            )
            # Use vocab_logits as a stand-in pooled rep (B, 1, d_model
            # is not exposed by the base model). To keep this faithful,
            # we instead pool the embeddings via a no-grad path on the
            # vocab_logits. The length head only needs a `(B, d_model)`
            # tensor, so we project from vocab space.
            pass
        # Pool over the sequence so length_head gets a (B, d_model) input.
        # We project by averaging the embeddings the base model produces
        # for `x_t`.
        emb = self.base.embed_tokens(x_t)
        if attention_mask is not None:
            denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (emb * attention_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = emb.mean(dim=1)
        length_logits = self.length_head(pooled)
        return vocab_logits, length_logits


class _ConfigureCtx:
    """Stand-in ``pl_module`` for :meth:`ILMLossWithMaskedCE.configure`.

    The real Harness only forwards ``self.device`` to ``configure`` here,
    so we provide just that attribute.
    """

    def __init__(self, device=torch.device("cpu")):
        self.device = device


class TestILMLossWithMaskedCEForward:
    """Exercise :meth:`ILMLossWithMaskedCE.__call__` end-to-end.

    Uses ``DefaultILMCollator`` + the shared ``real_loglinear_schedule``
    fixture to build a realistic sparse ``ILMBatch``, and wraps the base
    ILM model with a tiny length head so the loss-path assertion on
    ``length_logits`` passes.
    """

    @pytest.fixture()
    def num_classes(self):
        # Must be at least as large as ``total_n_drops + 1`` so the
        # ``length_loss="ce"`` branch (which feeds total_n_drops as a class
        # index into ``cross_entropy``) does not go out of bounds. The
        # collator below uses block_size=16, so 64 is a comfortable bound.
        return 64

    @pytest.fixture()
    def model(self, tiny_ilm_model, num_classes):
        return _LengthHeadWrapper(tiny_ilm_model, num_classes=num_classes)

    @pytest.fixture()
    def loss_fn(self, model, simple_tokenizer):
        loss = ILMLossWithMaskedCE(
            model=model,
            tokenizer=simple_tokenizer,
            length_loss="ce",
            length_loss_weight=0.5,
        )
        loss.configure(_ConfigureCtx())
        return loss

    @pytest.fixture()
    def batch(self, simple_tokenizer, real_loglinear_schedule):
        # The default ``_n_drop_uniformly`` samples from ``[0, seq_len]``,
        # so ``n_drops`` can be 0 — when that happens for every example,
        # ``n_drops_counts`` becomes an empty sparse tensor whose dtype
        # defaults to float (causing ``cross_entropy`` to reject the
        # target as non-Long). Force at least one drop per example to
        # keep this unit test deterministic.
        class _AtLeastOneDropCollator(DefaultILMCollator):
            sample_n_drops_fn = staticmethod(lambda seq_len: max(1, seq_len // 4))

        collator = _AtLeastOneDropCollator(
            tokenizer=simple_tokenizer,
            block_size=16,
            noise_schedule=real_loglinear_schedule,
            return_dense_target=False,
        )
        examples = [
            {
                "input_ids": torch.randint(
                    7, simple_tokenizer.vocab_size, (10,)
                ).tolist(),
            }
            for _ in range(2)
        ]
        return collator(examples)

    @pytest.mark.slow
    def test_loss_finite(self, loss_fn, batch):
        result = loss_fn(batch)
        assert "loss" in result
        assert torch.isfinite(result["loss"]).item(), (
            f"non-finite loss: {result['loss']}"
        )

    @pytest.mark.slow
    def test_loss_dict_shapes(self, loss_fn, batch):
        result = loss_fn(batch)
        n = batch["input_ids"].shape[0]
        for key in (
            "batch_loss",
            "per_example_length_loss",
            "per_example_ce",
            "n_drops",
        ):
            assert result[key].shape[0] == n, f"{key} has wrong batch dim"

    @pytest.mark.slow
    def test_loss_has_grad(self, model, loss_fn, batch):
        for p in model.parameters():
            p.grad = None
        result = loss_fn(batch)
        result["loss"].backward()
        # At least one model parameter should have a non-None gradient.
        assert any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in model.parameters()
        )
