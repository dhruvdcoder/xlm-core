"""Uniform canvas corruption for gemma-usdm.

Source tokens stay clean. A canvas token is replaced with probability ``t``
by a uniform draw from the vocabulary excluding the pad id.
"""

from __future__ import annotations

import torch
from torch import Tensor


def sample_uniform_except_pad(
    shape: torch.Size | tuple[int, ...],
    vocab_size: int,
    pad_id: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Uniform ids in ``[0, vocab_size)`` excluding ``pad_id``."""
    if vocab_size < 2:
        raise ValueError(
            f"vocab_size must be at least 2 so pad id {pad_id} can be excluded"
        )
    draws = torch.randint(
        0,
        vocab_size - 1,
        shape,
        device=device,
        generator=generator,
    )
    return draws + (draws >= pad_id).long()


def sample_time(
    batch_size: int,
    clean_prob: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Per-row noise level.

    With probability ``clean_prob`` the row is fully clean (``t = 0``).
    Otherwise ``t`` is uniform on ``(0, 1]``.
    """
    t = 1 - torch.rand(batch_size, device=device, generator=generator)
    if clean_prob <= 0:
        return t
    clean = (
        torch.rand(batch_size, device=device, generator=generator) < clean_prob
    )
    return torch.where(clean, torch.zeros_like(t), t)


def corrupt_canvas(
    clean_ids: Tensor,
    canvas_mask: Tensor,
    t: Tensor,
    vocab_size: int,
    pad_id: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Replace canvas tokens with probability ``t`` (shape ``(batch,)``).

    Positions where ``canvas_mask`` is false, including the source, are copied
    through. ``t = 0`` leaves the row unchanged.
    """
    if t.ndim != 1 or t.shape[0] != clean_ids.shape[0]:
        raise ValueError(
            f"t must have shape ({clean_ids.shape[0]},), got {tuple(t.shape)}"
        )
    unit = torch.rand(
        clean_ids.shape, device=clean_ids.device, generator=generator
    )
    replace = canvas_mask.to(dtype=torch.bool) & (unit < t.unsqueeze(-1))
    draws = sample_uniform_except_pad(
        clean_ids.shape,
        vocab_size,
        pad_id,
        clean_ids.device,
        generator,
    )
    return torch.where(replace, draws, clean_ids)
