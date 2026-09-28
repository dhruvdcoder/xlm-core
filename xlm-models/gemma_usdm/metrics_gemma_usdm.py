"""Reported metrics for gemma-usdm."""

from __future__ import annotations

from typing import Any, Dict

import torch


def nfe_update_fn(
    batch: Dict[str, Any],
    loss_dict: Dict[str, Any],
    tokenizer: Any = None,
) -> Dict[str, Any]:
    """Mean of the per-sequence denoising step counts."""
    del batch, tokenizer
    value = loss_dict["steps_taken"]
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.float32)
    else:
        value = value.detach().float()
    return {"value": value}
