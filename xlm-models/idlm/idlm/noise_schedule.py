"""Noise schedules for IDLM model.

This file implements different noise schedules used in IDLM diffusion training.
Copied from pcdd implementation and adapted for XLM framework.
"""

import torch
from typing import Tuple
from torch import Tensor as TT
from jaxtyping import Float
from xlm.noise import NoiseSchedule


def _convert_to_correlated(t: Float[TT, " *batch"]) -> Float[TT, " *batch"]:
    """Convert uniformly sampled batch of time indices to more evenly spaced samples."""
    device = t.device
    batch_size = t.shape[0]
    offset = (
        torch.arange(batch_size, device=device) / batch_size
    )  # values like 0/bs, 1/bs, 2/bs, ..., (bs-1)/bs
    # t/bs will be between 0 and 1/bs
    return (t / batch_size + offset) % 1  # %1 is only for numerical stability


class PoissonNoiseSchedule(NoiseSchedule):
    """Poisson noise schedule for IDLM."""

    antithetic_sampling: bool

    def __init__(
        self,
        sigma: float,
        c: float = 0.0,  # dirac delta weight at t=0
        antithetic_sampling: bool = False,
        eps: float = 1e-3,
    ):
        """Initialize Poisson noise schedule.

        Args:
            sigma: Noise rate parameter.
            c: Dirac delta weight at t=0.
            antithetic_sampling: Whether to use antithetic sampling.
            eps: Small epsilon to avoid t=0.
        """
        self.antithetic_sampling = antithetic_sampling
        self.sigma = sigma
        self.c = c
        self.eps = eps

    def noise_rate(self, t: Float[TT, " *batch"]) -> Float[TT, " *batch"]:
        return torch.full_like(t, self.sigma)

    def total_noise(self, t: Float[TT, " *batch"]) -> Float[TT, " *batch"]:
        return self.sigma * t + torch.full_like(t, self.c)

    def t_from_noise_rate(self, noise_rate: float) -> float:
        raise NotImplementedError(
            "Poisson noise schedule does not support t_from_noise_rate"
        )

    def t_from_total_noise(
        self, total_noise: Float[TT, " *batch"]
    ) -> Float[TT, " *batch"]:
        raise NotImplementedError(
            "Poisson noise schedule does not support t_from_total_noise"
        )

    def forward(
        self, t: Float[TT, " *batch"]
    ) -> Tuple[Float[TT, " *batch"], Float[TT, " *batch"]]:
        return self.noise_rate(t), self.total_noise(t)

    def __call__(
        self, t: Float[TT, " *batch"]
    ) -> Tuple[Float[TT, " *batch"], Float[TT, " *batch"]]:
        return self.forward(t)

    def sample_t(self, batch_size: int) -> Float[TT, " batch_size"]:
        t = torch.rand(batch_size)
        if self.antithetic_sampling:
            t = _convert_to_correlated(t)
        # keep samples away from 0 i.e start from 0+eps
        t = (1.0 - self.eps) * t + self.eps
        return t

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PoissonNoiseSchedule):
            return False
        return self.sigma == other.sigma


class LogLinearNoiseSchedule(NoiseSchedule):
    """Log-linear noise schedule for IDLM."""

    antithetic_sampling: bool

    def __init__(
        self,
        sigma_max: float,
        sigma_min: float = 1e-3,
        antithetic_sampling: bool = False,
        eps: float = 1e-3,
    ):
        """Initialize log-linear noise schedule.

        Args:
            sigma_max: Maximum noise rate.
            sigma_min: Minimum noise rate.
            antithetic_sampling: Whether to use antithetic sampling.
            eps: Small epsilon to avoid t=0.
        """
        self.antithetic_sampling = antithetic_sampling
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.eps = eps

    def noise_rate(self, t: Float[TT, " *batch"]) -> Float[TT, " *batch"]:
        sigma_max = torch.full_like(t, self.sigma_max)
        sigma_min = torch.full_like(t, self.sigma_min)
        return (
            sigma_min
            * torch.log(sigma_max / sigma_min)
            * (sigma_max / sigma_min) ** t
        )

    def total_noise(self, t: Float[TT, " *batch"]) -> Float[TT, " *batch"]:
        sigma_max = torch.full_like(t, self.sigma_max)
        sigma_min = torch.full_like(t, self.sigma_min)
        frac = sigma_max / sigma_min
        return (sigma_min / torch.log(frac)) * (frac**t - 1)

    def t_from_noise_rate(self, noise_rate: float) -> float:
        raise NotImplementedError(
            "LogLinear noise schedule does not support t_from_noise_rate"
        )

    def t_from_total_noise(
        self, total_noise: Float[TT, " *batch"]
    ) -> Float[TT, " *batch"]:
        raise NotImplementedError(
            "LogLinear noise schedule does not support t_from_total_noise"
        )

    def forward(
        self, t: Float[TT, " *batch"]
    ) -> Tuple[Float[TT, " *batch"], Float[TT, " *batch"]]:
        return self.noise_rate(t), self.total_noise(t)

    def __call__(
        self, t: Float[TT, " *batch"]
    ) -> Tuple[Float[TT, " *batch"], Float[TT, " *batch"]]:
        return self.forward(t)

    def sample_t(self, batch_size: int) -> Float[TT, " batch_size"]:
        t = torch.rand(batch_size)
        if self.antithetic_sampling:
            t = _convert_to_correlated(t)
        # keep samples away from 0 i.e start from 0+eps
        t = (1.0 - self.eps) * t + self.eps
        return t

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogLinearNoiseSchedule):
            return False
        return (
            self.sigma_max == other.sigma_max
            and self.sigma_min == other.sigma_min
        )


class GeometricNoiseSchedule(NoiseSchedule):
    """Geometric noise schedule for IDLM.

    The noise rate is the same as loglinear except we place a dirac delta at t=0 with weight sigma_min.
    This is just some mathematical trickery to avoid total_noise from being 0 at t=0.
    """

    antithetic_sampling: bool

    def __init__(
        self,
        sigma_max: float,
        sigma_min: float = 1e-5,
        antithetic_sampling: bool = False,
        eps: float = 1e-3,
    ):
        """Initialize geometric noise schedule.

        Args:
            sigma_max: Maximum noise rate.
            sigma_min: Minimum noise rate.
            antithetic_sampling: Whether to use antithetic sampling.
            eps: Small epsilon to avoid t=0.
        """
        self.antithetic_sampling = antithetic_sampling
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.eps = eps

    def noise_rate(self, t: Float[TT, " *batch"]) -> Float[TT, " *batch"]:
        sigma_min = torch.full_like(t, self.sigma_min)
        sigma_max = torch.full_like(t, self.sigma_max)
        return (
            sigma_max**t
            * sigma_min ** (1 - t)
            * torch.log(sigma_max / sigma_min)
        )

    def total_noise(self, t: Float[TT, " *batch"]) -> Float[TT, " *batch"]:
        sigma_min = torch.full_like(t, self.sigma_min)
        sigma_max = torch.full_like(t, self.sigma_max)
        return sigma_min * (sigma_max / sigma_min) ** t

    def t_from_noise_rate(self, noise_rate: float) -> float:
        raise NotImplementedError(
            "Geometric noise schedule does not support t_from_noise_rate"
        )

    def t_from_total_noise(
        self, total_noise: Float[TT, " *batch"]
    ) -> Float[TT, " *batch"]:
        raise NotImplementedError(
            "Geometric noise schedule does not support t_from_total_noise"
        )

    def forward(
        self, t: Float[TT, " *batch"]
    ) -> Tuple[Float[TT, " *batch"], Float[TT, " *batch"]]:
        return self.noise_rate(t), self.total_noise(t)

    def __call__(
        self, t: Float[TT, " *batch"]
    ) -> Tuple[Float[TT, " *batch"], Float[TT, " *batch"]]:
        return self.forward(t)

    def sample_t(self, batch_size: int) -> Float[TT, " batch_size"]:
        t = torch.rand(batch_size)
        if self.antithetic_sampling:
            t = _convert_to_correlated(t)
        # keep samples away from 0 i.e start from 0+eps
        t = (1.0 - self.eps) * t + self.eps
        return t
