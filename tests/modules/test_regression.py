"""Unit tests for :class:`xlm.modules.regression.RateHead`."""

import pytest
import torch

from xlm.modules.regression import RateHead


class TestRateHeadContinuous:
    def test_softplus_shape_and_bounds(self):
        head = RateHead(
            d_model=8,
            head_type="continuous",
            scalar_fn="softplus",
            min_val=0.0,
            max_val=2.0,
        )
        x = torch.randn(2, 4, 8)
        out = head(x)
        assert out.shape == (2, 4)
        # softplus output is positive and rescaled into [min_val, max_val).
        assert (out >= 0.0).all()

    def test_softplus_init_bias_keeps_output_near_min(self):
        head = RateHead(
            d_model=8,
            head_type="continuous",
            scalar_fn="softplus",
            init_bias=-10.0,
            min_val=0.0,
            max_val=1.0,
        )
        x = torch.zeros(2, 4, 8)
        out = head(x)
        # With a strongly negative bias, softplus(s) ~ 0 -> out ~ min_val
        assert (out < 0.05).all()

    def test_identity_output_unbounded(self):
        head = RateHead(
            d_model=8, head_type="continuous", scalar_fn="identity"
        )
        x = torch.randn(2, 3, 8)
        out = head(x)
        assert out.shape == (2, 3)

    def test_exp_output_positive(self):
        head = RateHead(d_model=8, head_type="continuous", scalar_fn="exp")
        x = torch.randn(2, 3, 8)
        out = head(x)
        assert (out >= 0.0).all()

    def test_sigmoid_output_in_range(self):
        head = RateHead(
            d_model=8,
            head_type="continuous",
            scalar_fn="sigmoid",
            min_val=0.0,
            max_val=1.0,
        )
        x = torch.randn(2, 3, 8)
        out = head(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_unknown_scalar_fn_raises(self):
        with pytest.raises(ValueError, match="scalar_fn"):
            RateHead(
                d_model=8, head_type="continuous", scalar_fn="weird"
            )


class TestRateHeadDiscretized:
    def test_output_in_range(self):
        head = RateHead(
            d_model=8,
            head_type="discretized",
            num_bins=4,
            min_val=0.0,
            max_val=10.0,
        )
        x = torch.randn(2, 3, 8)
        out = head(x)
        assert out.shape == (2, 3)
        assert (out >= 0.0).all() and (out <= 10.0).all()

    def test_bin_centers_registered(self):
        head = RateHead(
            d_model=4, head_type="discretized", num_bins=5, min_val=0.0, max_val=1.0
        )
        assert head.bin_centers.shape == (5,)
        assert torch.allclose(head.bin_centers[0], torch.tensor(0.0))
        assert torch.allclose(head.bin_centers[-1], torch.tensor(1.0))


class TestRateHeadConditioning:
    def test_with_conditioning_preserves_shape(self):
        head = RateHead(
            d_model=8,
            head_type="continuous",
            scalar_fn="sigmoid",
            cond_dim=4,
        )
        x = torch.randn(2, 3, 8)
        c = torch.randn(2, 4)
        out = head(x, c)
        assert out.shape == (2, 3)

    def test_gradient_reaches_adaln_modulation(self):
        head = RateHead(
            d_model=8,
            head_type="continuous",
            scalar_fn="exp",
            cond_dim=4,
        )
        x = torch.randn(2, 3, 8)
        c = torch.randn(2, 4, requires_grad=False)
        # AdaLN modulation parameters start at zero so they do not affect
        # the forward output, but the backward pass must still wire them up.
        out = head(x, c)
        out.sum().backward()
        # Find the modulation linear layer and check it received a gradient.
        adamod = head.adaLN_modulation.modulation
        assert adamod.weight.grad is not None


class TestRateHeadValidation:
    def test_unknown_head_type_raises(self):
        with pytest.raises(ValueError, match="head_type"):
            RateHead(d_model=8, head_type="banana")
