# v2
from typing import Protocol, Tuple
from torch import Tensor as TT
from jaxtyping import Float


class NoiseSchedule(Protocol):
    def noise_rate(self, t: Float[TT, " *batch"]) -> Float[TT, " *batch"]: ...

    def total_noise(self, t: Float[TT, " *batch"]) -> Float[TT, " *batch"]: ...

    def t_from_noise_rate(self, noise_rate: float) -> float: ...

    def t_from_total_noise(self, total_noise: float) -> float: ...

    def sample_t(self, batch_size: int) -> Float[TT, " batch_size"]: ...

    def forward(
        self, t: Float[TT, " *batch"]
    ) -> Tuple[Float[TT, " *batch"], Float[TT, " *batch"]]: ...

    def __call__(
        self, t: Float[TT, " *batch"]
    ) -> Tuple[Float[TT, " *batch"], Float[TT, " *batch"]]: ...


class DummyNoiseSchedule(NoiseSchedule):
    def noise_rate(self, t: Float[TT, "batch"]) -> Float[TT, "batch"]:
        raise NotImplementedError("This noise schedule does nothing.")

    def total_noise(self, t: Float[TT, "batch"]) -> Float[TT, "batch"]:
        raise NotImplementedError("This noise schedule does nothing.")

    def t_from_noise_rate(self, noise_rate: float) -> float:
        raise NotImplementedError("This noise schedule does nothing.")

    def t_from_total_noise(self, total_noise: float) -> float:
        raise NotImplementedError("This noise schedule does nothing.")

    def sample_t(self, batch_size: int) -> Float[TT, "batch_size"]:
        raise NotImplementedError("This noise schedule does nothing.")

    def forward(
        self, t: Float[TT, " *batch"]
    ) -> Tuple[Float[TT, " *batch"], Float[TT, " *batch"]]:
        raise NotImplementedError("This noise schedule does nothing.")

    def __call__(
        self, t: Float[TT, " *batch"]
    ) -> Tuple[Float[TT, " *batch"], Float[TT, " *batch"]]:
        raise NotImplementedError("This noise schedule does nothing.")
