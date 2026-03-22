import abc
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor


class Schedule(abc.ABC):
    """
    Generic schedule class for masking or noising
    This represents function a : [0, 1] -> [0, 1] satisfying a(0) = 0, a(1) = 1 or at least approximately
    """

    @abc.abstractmethod
    def at(self, t: Tensor):
        """
        Return value a(t)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def derivative_at(self, t: Tensor):
        """
        Return d/dt a(t)
        """
        raise NotImplementedError

    def rate_scale_factor(self, t: Tensor) -> Tensor:
        """
        Return d/dt a(t) / (1 - a(t)) common in rate matrix calculation
        """
        return self.derivative_at(t) / (1 - self.at(t))

    def sample(self, shape, device) -> Tensor:
        """
        Sample from the schedule, returns a tensor of shape `shape` with values in [0, 1]
        """
        uniform = torch.rand(shape, device=device)
        return self.inv(uniform)

    def sample_truncated(self, threshold, shape, device) -> Tensor:
        """
        Sample from a truncated schedule, returns a tensor of shape `shape` with values in [threshold, 1]
        """
        uniform = torch.rand(shape, device=device)
        threshold = self.at(threshold)
        return self.inv(uniform * (1 - threshold) + threshold)

    @abc.abstractmethod
    def inv(self, alpha: Tensor):
        """
        Given alpha in [0, 1] such that a(t)=alpha, returns the corresponding t.
        """
        raise NotImplementedError


class LinearSchedule(Schedule):
    def __init__(self):
        pass

    def at(self, t: Tensor):
        return t

    def derivative_at(self, t: Tensor):
        return torch.ones_like(t, device=t.device)

    def inv(self, alpha: Tensor):
        return alpha


class GeometricSchedule(Schedule, nn.Module):
    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        self.register_buffer("min", Tensor([min_val]))
        self.register_buffer("max", Tensor([max_val]))

    def at(self, t: Tensor):
        min_val = self.min.to(t.device)
        max_val = self.max.to(t.device)
        return torch.exp(-(min_val ** (1 - t)) * max_val**t)

    def derivative_at(self, t):
        min_val = self.min.to(t.device)
        max_val = self.max.to(t.device)
        return (
            self.at(t)
            * min_val ** (1 - t)
            * max_val**t
            * (min_val.log() - max_val.log())
        )

    def inv(self, alpha: Tensor):
        log_min = self.min.to(alpha.device).log()
        log_max = self.max.to(alpha.device).log()
        return (torch.log(-torch.log(alpha)) - log_min) / (log_max - log_min)


class SinSchedule(Schedule, nn.Module):
    def __init__(self):
        super().__init__()

    def at(self, t: Tensor):
        return torch.sin(torch.pi / 2 * t)

    def derivative_at(self, t: Tensor):
        return (torch.pi / 2) * torch.cos(torch.pi / 2 * t)

    def inv(self, alpha: Tensor):
        return (2 / torch.pi) * torch.asin(alpha.clamp(min=0., max=1.))


class CosineSchedule(Schedule, nn.Module):
    def __init__(self):
        super().__init__()

    def at(self, t: Tensor):
        return 1 - torch.cos(torch.pi / 2 * t)
    
    def derivative_at(self, t: Tensor):
        return (torch.pi / 2) * torch.sin(torch.pi / 2 * t)
    
    def rate_scale_factor(self, t):
        return (torch.pi/2) * torch.tan(torch.pi / 2 * t)
    
    def inv(self, alpha):
        return (2 / torch.pi) * torch.arccos(1 - alpha.clamp(min=0., max=1.))


class PolynomialSchedule(Schedule, nn.Module):
    def __init__(self, exp):
        super().__init__()
        self.exp = exp
    
    def at(self, t: Tensor):
        return t ** self.exp
    
    def derivative_at(self, t: Tensor):
        return self.exp * t ** (self.exp - 1)
    
    def inv(self, alpha: Tensor):
        return alpha ** (1 / self.exp)
    
    
class FlexMDMNoiseSchedule:
    def __init__(self, insertion_noise_schedule, unmasking_noise_schedule):
        self.insertion_noise_schedule = insertion_noise_schedule
        self.unmasking_noise_schedule = unmasking_noise_schedule
