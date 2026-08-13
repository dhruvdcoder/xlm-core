import torch
import torch.nn as nn
import abc

class Noise(abc.ABC, nn.Module):
  """
  Baseline forward method to get the total + rate of noise at a timestep
  """
  
  def forward(self, t):
    return self.compute_loss_scaling_and_move_chance(t)
  
class CosineNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def compute_loss_scaling_and_move_chance(self, t):
    cos = - (1 - self.eps) * torch.cos(t * torch.pi / 2)
    sin = - (1 - self.eps) * torch.sin(t * torch.pi / 2)
    move_chance = cos + 1
    loss_scaling = sin / (move_chance + self.eps) * torch.pi / 2
    return loss_scaling, move_chance

class ExpNoise(Noise):
  def __init__(self, exp=2, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.exp = exp
  
  def compute_loss_scaling_and_move_chance(self, t):
    move_chance = torch.pow(t, self.exp)
    move_chance = torch.clamp(move_chance, min=self.eps)
    loss_scaling = - (self.exp * torch.pow(t, self.exp-1)) / move_chance
    return loss_scaling, move_chance

class LogarithmicNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def compute_loss_scaling_and_move_chance(self, t):
    move_chance = torch.log1p(t) / torch.log(torch.tensor(2.0))
    loss_scaling = - 1 / (move_chance * torch.log(torch.tensor(2.0)) * (1 + t))
    return loss_scaling, move_chance

class LogLinearNoise(Noise):
  """Log Linear noise schedule.
  
  Built such that 1 - 1/e^(n(t)) interpolates between 0 and
  ~1 when t varies from 0 to 1. Total noise is
  -log(1 - (1 - eps) * t), so the sigma will be
  (1 - eps) * t.
  """
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.sigma_max = self.total_noise(torch.tensor(1.0))
    self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

  def rate_noise(self, t):
    return (1 - self.eps) / (1 - (1 - self.eps) * t)

  def total_noise(self, t):
    return -torch.log1p(-(1 - self.eps) * t)

  def compute_loss_scaling_and_move_chance(self, t):
    loss_scaling = - 1 / t
    return loss_scaling, t


class Bd3lmNoise(nn.Module):
    def __init__(
        self,
        block_size: int,
        model_length: int,
        antithetic_sampling: bool = True,
        sampling_eps_min: float = 1e-3,
        sampling_eps_max: float = 1.0,
        eps: float = 1e-3,
        noise_type: str = "loglinear",
        resample: bool = False,
    ):
      super().__init__()
      self.block_size = block_size
      self.model_length = model_length
      self.antithetic_sampling = antithetic_sampling
      self.sampling_eps_min = sampling_eps_min
      self.sampling_eps_max = sampling_eps_max
      self.eps = eps
      self.noise = self.get_noise(noise_type)
      self.resample = resample
    def get_noise(self, noise_type=None):

      if noise_type == 'loglinear':
        return LogLinearNoise(eps=self.eps)
      elif noise_type == 'square':
        return ExpNoise(2)
      elif noise_type == 'square_root':
        return ExpNoise(0.5)
      elif noise_type == 'log':
        return LogarithmicNoise()
      elif noise_type == 'cosine':
        return CosineNoise()
      else:
        raise ValueError(f'{noise_type} is not a valid noise')

    def _sample_t(
      self, batch_dims, device, sampling_eps_min = 1e-3, sampling_eps_max = 1.0, block_size=None):
      if block_size is None:
        block_size = self.block_size
      n = batch_dims[-1]
      num_blocks = n // block_size
      _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

      # antithetic sampling along blocks & batches (for uniform sampling)
      if self.antithetic_sampling:
        offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
        offset_b = offset_b.view(batch_dims[0], num_blocks)
        _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1
      t = _eps_b
      if block_size != self.model_length:
        t = t.repeat_interleave(block_size, dim=-1)
   
      # nll
      if self.sampling_eps_max >= 1 and self.sampling_eps_min >= 1:
        return torch.ones_like(t)
      t = t * (self.sampling_eps_max - self.sampling_eps_min) + self.sampling_eps_min
      return t
    
    def _sigma_from_p(self, p):
      return torch.min(- torch.log(1 - p), self.noise.sigma_max)
    
    def _resample_q_xt(
      self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max,mask_token_id, pad_token_id=None):
      """Resamples x_t if the percentage of masked tokens is outside the bounds
      defined by sampling_eps_min and sampling_eps_max.

      Blocks containing pad tokens are exempt from the bounds: with pads
      unmaskable they may never be able to reach the required masked fraction,
      so they keep their initial (pad-respecting) draw."""
      x_blocks = x.reshape(x.shape[0], -1, block_size)
      if pad_token_id is not None:
        eligible = ~(x_blocks == pad_token_id).any(-1)
      else:
        eligible = torch.ones(
          x_blocks.shape[:2], dtype=torch.bool, device=x.device)

      def _violations(perc_masked):
        v = torch.zeros_like(eligible)
        # if a bound is epsilon, don't resample for it
        if sampling_eps_min != 1e-3:
          v = v | (perc_masked < sampling_eps_min)
        if sampling_eps_max != 1:
          v = v | (perc_masked > sampling_eps_max)
        return v & eligible

      perc_masked = (xt == mask_token_id).float().sum(-1) / block_size
      regen_idx = _violations(perc_masked)
      while regen_idx.any():
        regen_flat = regen_idx.repeat_interleave(block_size, dim=-1)
        new_moves = torch.rand(* x.shape, device=x.device) < p
        if pad_token_id is not None:
          new_moves = new_moves & (x != pad_token_id)
        move_indices[regen_flat] = new_moves[regen_flat]
        xt = torch.where(move_indices, mask_token_id, x)
        xt = xt.reshape(xt.shape[0], -1, block_size)
        perc_masked = (xt == mask_token_id).float().sum(-1) / block_size
        regen_idx = _violations(perc_masked)
      return xt
  

    def q_xt(
      self, x, p,mask_token_id,pad_token_id, block_size=None, sampling_eps_min=None, sampling_eps_max=None):
      """Computes the noisy sample xt.

      Args:
        x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
        p: float torch.Tensor with shape (batch_size, 1).
        block_size: int, block size.
        sampling_eps_min: float, minimum percentage of masked tokens.
        sampling_eps_max: float, maximum percentage of masked tokens.
      """
    
      if block_size is None:
        block_size = self.block_size
  
      move_indices = torch.rand(
        * x.shape, device=x.device) <= p
      if pad_token_id is not None:
        move_indices = move_indices & (x != pad_token_id)
      xt = torch.where(move_indices, mask_token_id, x)

      if block_size == 1 and self.sampling_eps_min == 1.0:
        return torch.full_like(x, mask_token_id)
      
      # no need to resample for bounds 1e-3, 1
      if self.resample and \
        not (self.sampling_eps_min == 1e-3 and self.sampling_eps_max == 1.0):
        xt = xt.reshape(xt.shape[0], -1, block_size)
        xt = self._resample_q_xt(x,
                               xt,
                               move_indices,
                               p,
                               block_size,
                               self.sampling_eps_min,
                               self.sampling_eps_max,
                               mask_token_id,
                               pad_token_id=pad_token_id)
        xt = xt.reshape(xt.shape[0], -1)
      return xt

    def forward(self, t):
      return self.noise(t)
    