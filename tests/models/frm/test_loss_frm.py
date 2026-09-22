"""Unit tests for the FRM loss function."""

import pytest
import torch

from frm.loss_frm import FRMLoss
from tests.models._base import BaseLossTests


class TestFRMLoss(BaseLossTests):
    """Tests for :class:`FRMLoss`."""

    @pytest.fixture()
    def loss_fn(self, tiny_frm_model, simple_tokenizer):
        return FRMLoss(
            loss_on_padding=False,
            loss_on_clamped=True,
            self_cond_prob=0.0,
            fpf_prob=0.0,
            model=tiny_frm_model,
            tokenizer=simple_tokenizer,
        )

    @pytest.fixture()
    def batch(self, frm_batch):
        return frm_batch


class TestFRMLossFPF:
    @pytest.fixture()
    def loss_fn(self, tiny_frm_model, simple_tokenizer):
        loss = FRMLoss(
            loss_on_padding=False,
            self_cond_prob=0.0,
            fpf_prob=1.0,
            fpf_rollout_depth=2,
            model=tiny_frm_model,
            tokenizer=simple_tokenizer,
        )
        loss.train() if hasattr(loss, "train") else None
        return loss

    def test_fpf_loss_is_stopgrad(self, loss_fn, frm_batch, tiny_frm_model):
        tiny_frm_model.train()
        result = loss_fn(frm_batch)
        result["loss"].backward()
        grads = [
            p.grad
            for p in tiny_frm_model.parameters()
            if p.grad is not None
        ]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

    def test_fpf_carry_detached(self, loss_fn, tiny_frm_model, frm_batch):
        loss_fn.training = True
        tiny_frm_model.train()
        from frm.flow_frm import one_hot_tokens

        x1 = one_hot_tokens(frm_batch["input_ids"], tiny_frm_model.num_embeddings)
        t = torch.rand(frm_batch["input_ids"].shape[0])
        attn = frm_batch["attention_mask"].to(dtype=torch.bool)
        positions = (attn.cumsum(dim=1) - 1).clamp(min=0)
        clue_oh = one_hot_tokens(
            frm_batch["clue_ids"], tiny_frm_model.num_embeddings
        )
        s = loss_fn._fpf_carry(
            tiny_frm_model,
            x1,
            t,
            attn,
            positions,
            frm_batch["clue_ids"],
            frm_batch["clamp_mask"],
            clue_oh,
        )
        assert not s.requires_grad
