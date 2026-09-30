"""Continuous interpolant helpers for Flow Reasoning Models."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor as TT

INTERPOLANT_EPS = 1e-5


def one_hot_tokens(ids: TT, vocab_size: int) -> TT:
    """Convert token ids to float one-hots; negative ids (ignore) map to 0."""
    safe = ids.clamp(min=0).long()
    return F.one_hot(safe, num_classes=vocab_size).to(dtype=torch.float32)


def simplex_noise(like: TT) -> TT:
    """Random point on the probability simplex (softmax of Gaussian)."""
    return F.softmax(torch.randn_like(like), dim=-1)


def interpolant(x1: TT, t: TT, noise: Optional[TT] = None) -> TT:
    """``x_t = (1-t) ε + t x_1`` with ``ε`` on the simplex and ``t`` of shape ``(batch,)``."""
    if noise is None:
        noise = simplex_noise(x1)
    t_ = t.to(dtype=x1.dtype).view(-1, 1, 1)
    return (1.0 - t_) * noise + t_ * x1


def velocity(pred_clean: TT, x_t: TT, t: TT) -> TT:
    """``v = (softmax(D) - x_t) / (1-t)``."""
    denom = (1.0 - t.to(dtype=x_t.dtype)).clamp(min=INTERPOLANT_EPS).view(
        -1, 1, 1
    )
    return (pred_clean - x_t) / denom


def apply_clamp(x: TT, clue_onehot: TT, clamp_mask: Optional[TT]) -> TT:
    """Overwrite clamped positions with the clue one-hot."""
    if clamp_mask is None:
        return x
    mask = clamp_mask.to(dtype=torch.bool).unsqueeze(-1)
    return torch.where(mask, clue_onehot.to(dtype=x.dtype), x)


def expand_t(t: TT, batch: int, device: torch.device) -> TT:
    if t.ndim == 0:
        return t.expand(batch).to(device=device)
    return t.to(device=device)
