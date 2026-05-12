"""Unit tests for :mod:`mdlm.noise_mdlm`."""

import pytest
import torch

from mdlm.noise_mdlm import (
    ContinuousTimeLinearSchedule,
    ContinuousTimeLogLinearSchedule,
    _convert_to_correlated,
)


class TestConvertToCorrelated:
    def test_output_in_unit_interval(self):
        t = torch.rand(8)
        out = _convert_to_correlated(t)
        assert out.shape == (8,)
        assert (out >= 0.0).all()
        assert (out < 1.0).all()

    def test_constant_input_is_evenly_spread(self):
        # When all t are the same, the offset puts them at i/bs.
        t = torch.zeros(4)
        out = _convert_to_correlated(t)
        assert torch.allclose(out, torch.tensor([0.0, 0.25, 0.5, 0.75]))


class TestContinuousTimeLinearSchedule:
    def test_total_noise_at_zero_is_zero(self):
        sched = ContinuousTimeLinearSchedule(sigma_min=0.5, sigma_max=2.0)
        out = sched.total_noise(torch.tensor([0.0]))
        assert torch.allclose(out, torch.tensor([0.0]))

    def test_t_from_total_noise_is_linear_inverse(self):
        # ``t_from_total_noise`` is the inverse of an *affine* (sigma_min +
        # (sigma_max - sigma_min) * t) parameterisation, not of the actual
        # ``total_noise`` (which is exponential). We check only the linear
        # inverse here, which is what the implementation provides.
        sched = ContinuousTimeLinearSchedule(sigma_min=0.5, sigma_max=2.0)
        affine = torch.tensor([0.5, 0.95, 1.4, 1.85])
        t = sched.t_from_total_noise(affine)
        # (affine - 0.5) / 1.5
        expected = (affine - 0.5) / (2.0 - 0.5)
        assert torch.allclose(t, expected, atol=1e-6)

    def test_t_from_noise_rate_raises(self):
        sched = ContinuousTimeLinearSchedule(sigma_min=0.0, sigma_max=1.0)
        with pytest.raises(RuntimeError):
            sched.t_from_noise_rate(torch.tensor([0.5]))

    def test_grad_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ContinuousTimeLinearSchedule(
                sigma_min=0.0, sigma_max=1.0, grad=True
            )

    def test_importance_sampling_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ContinuousTimeLinearSchedule(
                sigma_min=0.0, sigma_max=1.0, importance_sampling=True
            )


class TestContinuousTimeLogLinearSchedule:
    def test_total_noise_at_zero_is_zero(self):
        sched = ContinuousTimeLogLinearSchedule(sigma_min=0.0, sigma_max=2.0)
        out = sched.total_noise(torch.tensor([0.0]))
        assert torch.allclose(out, torch.tensor([0.0]), atol=1e-6)

    def test_noise_rate_positive(self):
        sched = ContinuousTimeLogLinearSchedule(sigma_min=0.0, sigma_max=2.0)
        rates = sched.noise_rate(torch.tensor([0.1, 0.5, 0.9]))
        assert (rates > 0.0).all()

    def test_round_trip_total_noise(self):
        sched = ContinuousTimeLogLinearSchedule(sigma_min=0.0, sigma_max=3.0)
        t = torch.tensor([0.05, 0.25, 0.5, 0.85])
        recovered = sched.t_from_total_noise(sched.total_noise(t))
        assert torch.allclose(recovered, t, atol=1e-5)

    def test_round_trip_noise_rate(self):
        sched = ContinuousTimeLogLinearSchedule(sigma_min=0.0, sigma_max=3.0)
        t = torch.tensor([0.05, 0.25, 0.5, 0.85])
        recovered = sched.t_from_noise_rate(sched.noise_rate(t))
        assert torch.allclose(recovered, t, atol=1e-5)

    def test_sigma_min_positive_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ContinuousTimeLogLinearSchedule(sigma_min=0.1, sigma_max=2.0)


class TestSampleT:
    def test_shape_and_bounds(self):
        sched = ContinuousTimeLinearSchedule(
            sigma_min=0.0, sigma_max=1.0, antithetic_sampling=True, eps=1e-3
        )
        t = sched.sample_t(batch_size=8)
        assert t.shape == (8,)
        assert (t >= 1e-3).all()
        assert (t <= 1.0).all()

    def test_no_antithetic_branch(self):
        sched = ContinuousTimeLinearSchedule(
            sigma_min=0.0, sigma_max=1.0, antithetic_sampling=False, eps=1e-3
        )
        t = sched.sample_t(batch_size=4)
        assert t.shape == (4,)
        assert (t >= 1e-3).all()
        assert (t <= 1.0).all()
