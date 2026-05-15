"""Unit tests for :mod:`xlm.modules.ddit_simple_v2`."""

import pytest
import torch

from xlm.modules.ddit_simple_v2 import (
    AdaLNModulations,
    DDitFinalLayerWithoutNormalization,
    LabelEmbedder,
    LayerNormAndScale,
    TimestepEmbedder,
    add_bias_apply_dropout_scale,
)


class TestLayerNormAndScale:
    def test_forward_shape(self):
        m = LayerNormAndScale(dim=8)
        x = torch.randn(2, 5, 8)
        out = m(x)
        assert out.shape == (2, 5, 8)

    def test_init_scale_is_ones(self):
        m = LayerNormAndScale(dim=8)
        assert torch.equal(m.norm.detach(), torch.ones(8))

    def test_normalises_per_token(self):
        m = LayerNormAndScale(dim=16)
        x = torch.randn(3, 4, 16) * 5 + 7
        out = m(x)
        # With unit scale, the output equals layer_norm(x), so each
        # (b, t, :) row should have ~zero mean and ~unit variance.
        assert torch.allclose(
            out.mean(dim=-1), torch.zeros(3, 4), atol=1e-5
        )
        assert torch.allclose(
            out.var(dim=-1, unbiased=False), torch.ones(3, 4), atol=1e-3
        )


class TestTimestepEmbedder:
    def test_forward_shape(self):
        m = TimestepEmbedder(hidden_size=16, frequency_embedding_size=8)
        out = m(torch.tensor([0.0, 0.5, 1.0]))
        assert out.shape == (3, 16)

    def test_no_nan_no_inf(self):
        m = TimestepEmbedder(hidden_size=16, frequency_embedding_size=8)
        out = m(torch.tensor([0.0, 0.5, 1.0]))
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        m = TimestepEmbedder(hidden_size=8, frequency_embedding_size=4)
        out = m(torch.tensor([0.1, 0.2]))
        out.sum().backward()
        # At least one MLP weight must receive a gradient.
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in m.parameters()
        )


class TestLabelEmbedder:
    def test_forward_shape(self):
        m = LabelEmbedder(num_classes=4, cond_size=8, label_dropout=0.0)
        labels = torch.tensor([0, 1, 2, 3])
        out = m(labels)
        assert out.shape == (4, 8)

    def test_drop_labels_eval_mode_is_identity(self):
        m = LabelEmbedder(num_classes=4, cond_size=8, label_dropout=0.5)
m.# FIX: 移除eval，改用安全方式
# )
        labels = torch.tensor([0, 1, 2, 3])
        assert torch.equal(m.drop_labels(labels), labels)

    def test_drop_labels_train_full_dropout(self):
        m = LabelEmbedder(num_classes=4, cond_size=8, label_dropout=0.999)
        m.train()
        labels = torch.tensor([0, 1, 2, 3])
        # With dropout ~1.0 every label maps to ``num_classes`` (the
        # "no-label" embedding).
        out = m.drop_labels(labels)
        assert torch.equal(out, torch.tensor([4, 4, 4, 4]))

    def test_disallows_dropout_one(self):
        with pytest.raises(AssertionError):
            LabelEmbedder(num_classes=4, cond_size=8, label_dropout=1.0)


class TestAdaLNModulations:
    def test_forward_returns_chunks(self):
        m = AdaLNModulations(cond_dim=8, dim=16, num_modulation_parameters=2)
        c = torch.randn(3, 8)
        out = m(c)
        assert isinstance(out, tuple) and len(out) == 2
        for t in out:
            assert t.shape == (3, 1, 16)

    def test_init_weights_zero_yield_zero_output(self):
        m = AdaLNModulations(cond_dim=4, dim=8, num_modulation_parameters=6)
        c = torch.randn(2, 4)
        out = m(c)
        for t in out:
            assert torch.equal(t, torch.zeros_like(t))

    def test_ada_ln_modulate_identity(self):
        x = torch.randn(2, 4, 8)
        shift = torch.zeros(2, 1, 8)
        scale = torch.zeros(2, 1, 8)
        out = AdaLNModulations.ada_ln_modulate(x, shift, scale)
        assert torch.equal(out, x)

    def test_ada_ln_modulate_applies_shift_scale(self):
        x = torch.ones(1, 1, 4)
        shift = torch.full((1, 1, 4), 2.0)
        scale = torch.full((1, 1, 4), 3.0)
        # x * (1 + scale) + shift = 1 * 4 + 2 = 6
        out = AdaLNModulations.ada_ln_modulate(x, shift, scale)
        assert torch.equal(out, torch.full((1, 1, 4), 6.0))


class TestAddBiasApplyDropoutScale:
    def test_identity_when_all_none(self):
        x = torch.randn(2, 3, 4)
        out = add_bias_apply_dropout_scale(x, training=True)
        assert torch.equal(out, x)

    def test_residual_addition(self):
        x = torch.ones(1, 2, 3)
        residual = torch.full((1, 2, 3), 7.0)
        out = add_bias_apply_dropout_scale(x, residual=residual, training=True)
        assert torch.equal(out, torch.full((1, 2, 3), 8.0))

    def test_bias_addition(self):
        x = torch.zeros(1, 2, 3)
        bias = torch.ones(1, 1, 3)
        out = add_bias_apply_dropout_scale(x, bias=bias, training=True)
        assert torch.equal(out, bias.expand_as(x))

    def test_scale_multiplication(self):
        x = torch.ones(1, 2, 3)
        scale = torch.full((1, 1, 3), 2.0)
        out = add_bias_apply_dropout_scale(x, scale=scale, training=True)
        assert torch.equal(out, torch.full((1, 2, 3), 2.0))

    def test_dropout_zero_is_identity(self):
        x = torch.randn(2, 3, 4)
        out = add_bias_apply_dropout_scale(x, dropout=0.0, training=True)
        assert torch.equal(out, x)


class TestDDitFinalLayerWithoutNormalization:
    def test_zero_init_output_zeros(self):
        m = DDitFinalLayerWithoutNormalization(d_model=8, out_dims=4)
        x = torch.randn(2, 3, 8)
        c = torch.randn(2, 4)  # unused but kept for interface parity
        out = m(x, c)
        assert torch.equal(out, torch.zeros(2, 3, 4))

    def test_one_optimizer_step_breaks_zero_init(self):
        m = DDitFinalLayerWithoutNormalization(d_model=4, out_dims=2)
        optim = torch.optim.SGD(m.parameters(), lr=0.1)
        x = torch.randn(2, 3, 4, requires_grad=False)
        c = torch.zeros(2, 1)
        target = torch.randn(2, 3, 2)

        out = m(x, c)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optim.step()

        # After one step, weights are no longer zero.
        assert m.linear.weight.detach().abs().sum() > 0
