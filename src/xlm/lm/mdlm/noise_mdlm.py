from typing import (
    Tuple,
)
import torch
from jaxtyping import Float
from torch import Tensor as TT
from xlm.utils.rank_zero import RankedLogger
from xlm.noise import NoiseSchedule

logger = RankedLogger(__name__, rank_zero_only=True)


def _convert_to_correlated(t: Float[TT, " batch"]) -> Float[TT, " batch"]:
    """
    Convert a uniformly sampled batch of time indices in [0,1] to a batch of more evenly spaced sample in [0,1].
    """
    device = t.device
    batch_size = t.shape[0]
    offset = (
        torch.arange(batch_size, device=device) / batch_size
    )  # values like 0/bs, 1/bs, 2/bs, ..., (bs-1)/bs
    # t/bs will be between 0 and 1/bs
    return (t / batch_size + offset) % 1  # %1 is only for numerical stability


class ContinousTimeNoiseSchedule(torch.nn.Module, NoiseSchedule):
    """Base class for continuous time noise schedules for absorbing diffusion.

    For absorbing diffusion in continuous time, we only need $\sigma(t)$ and
    the integral $\dot\sigma(t)$, which we call noise_rate and total_noise respectively.

    Note:
        We assume that for continous time, $t \in [0, 1]$.
    """

    def __init__(
        self,
        antithetic_sampling: bool = True,
        importance_sampling: bool = False,
        grad: bool = False,
        eps: float = 1e-3,
    ):
        """
        Args:
            antithetic_sampling: If true, the sampled time steps in a batch are
                sampled around points of a uniform grid over [0, 1], insted of sampling
                directly from a uniform distribution over [0, 1].

            importance_sampling: The goal is have a desired distribution over the noise level
                $\sigma$, sampling of $t$ is just a way of obtaining a value of $\sigma$.
                Since $\sigma(t)$ is non-linear function of $t$, if we want to have a
                desired distribution over $\sigma$ for training, which is indeed the case,
                we cannot simply sample $t$ uniformly and then transform it to $\sigma(t)$.
                Setting importance_sampling=True, will sample uniformly directly over $\sigma$
                in the range $[\sigma_{\text{min}}, \sigma_{\text{max}}]$.
        """
        super().__init__()
        if grad:
            raise NotImplementedError(
                "Gradient computation is not implemented"
            )
        self.antithetic_sampling = antithetic_sampling
        if antithetic_sampling:
            logger.info(
                "Antithetic sampling is enabled in the noise schedule."
            )
        self.importance_sampling = importance_sampling
        if importance_sampling:
            # TODO (importance sampling): Implement importance sampling
            raise NotImplementedError("Importance sampling is not implemented")
        self.grad = grad
        self.eps = eps

    def noise_rate(self, t: Float[TT, " batch"]) -> Float[TT, " batch"]:
        """Return the noise level at time t."""
        raise NotImplementedError

    def total_noise(self, t: Float[TT, " batch"]) -> Float[TT, " batch"]:
        """Return the total noise at time t."""
        raise NotImplementedError

    def t_from_noise_rate(
        self, noise_rate: Float[TT, " batch"]
    ) -> Float[TT, " batch"]:
        """Return the time step t from the noise level sigma."""
        raise NotImplementedError

    def t_from_total_noise(
        self, total_noise: Float[TT, " batch"]
    ) -> Float[TT, " batch"]:
        """Return the time step t from the total noise."""
        raise NotImplementedError

    def forward(
        self, t: Float[TT, " batch"]
    ) -> Tuple[Float[TT, " batch"], Float[TT, " batch"]]:
        """Return the noise level at time t.

        Args:
            t: The time step tensor of shape (batch)
        Returns:
            The noise level tensor of shape (batch)
            The total noise tensor of shape (batch)
        """
        return self.noise_rate(t), self.total_noise(t)

    def sample_t(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> Float[TT, " batch"]:
        """Sample a t uniformly from [0, 1]."""
        # if flags.DEBUG_OVERFIT:
        #    return torch.ones(batch_size, device=device) * 0.5
        t = torch.rand(
            (batch_size,), device=device
        )  # sample from a uniform distribution over [0, 1]
        if self.antithetic_sampling:
            t = _convert_to_correlated(t)
        # keep samples away from 0
        t = (1 - self.eps) * t + self.eps
        return t


class ContinuousTimeLinearSchedule(ContinousTimeNoiseSchedule):
    sigma_min: torch.Tensor
    sigma_max: torch.Tensor

    def __init__(self, sigma_min: float, sigma_max: float, **kwargs):
        super().__init__(**kwargs)
        if self.grad:
            self.sigma_min = torch.nn.Parameter(torch.tensor(sigma_min))
            self.sigma_max = torch.nn.Parameter(torch.tensor(sigma_max))
        else:
            self.register_buffer("sigma_min", torch.tensor(sigma_min))
            self.register_buffer("sigma_max", torch.tensor(sigma_max))

    def noise_rate(self, t: Float[TT, " batch"]) -> Float[TT, " batch"]:
        return (
            self.sigma_max - self.sigma_min
        )  # BUG: Needs to take the shape of t

    def total_noise(self, t: Float[TT, " batch"]) -> Float[TT, " batch"]:
        return self.sigma_min * torch.exp(self.sigma_max * t) - self.sigma_min

    def t_from_noise_rate(
        self, noise_rate: Float[TT, " batch"]
    ) -> Float[TT, " batch"]:
        raise RuntimeError(
            "Cannot compute t from noise rate for continuous time linear schedule"
        )

    def t_from_total_noise(
        self, total_noise: Float[TT, " batch"]
    ) -> Float[TT, " batch"]:
        return (total_noise - self.sigma_min) / (
            self.sigma_max - self.sigma_min
        )


class ContinuousTimeLogLinearSchedule(ContinuousTimeLinearSchedule):
    def __init__(self, sigma_min: float, sigma_max: float, **kwargs):
        super().__init__(sigma_min, sigma_max, **kwargs)
        if self.grad:
            raise NotImplementedError(
                "Gradient computation for loglinear schedule is not implemented"
            )
        if bool((self.sigma_min > 0)):
            raise NotImplementedError(
                "support for sigma_min > 0 is not implemented for loglinear schedule"
            )
        self.one_minus_eps = 1.0 - torch.exp(-self.sigma_max)

    def noise_rate(self, t: Float[TT, " batch"]) -> Float[TT, " batch"]:
        return self.one_minus_eps / (1.0 - self.one_minus_eps * t)

    def total_noise(self, t: Float[TT, " batch"]) -> Float[TT, " batch"]:
        return -torch.log1p(-self.one_minus_eps * t)

    def t_from_noise_rate(
        self, noise_rate: Float[TT, " batch"]
    ) -> Float[TT, " batch"]:
        return (1.0 / self.one_minus_eps) - (1.0 / noise_rate)

    def t_from_total_noise(
        self, total_noise: Float[TT, " batch"]
    ) -> Float[TT, " batch"]:
        return -torch.expm1(-total_noise) / self.one_minus_eps
