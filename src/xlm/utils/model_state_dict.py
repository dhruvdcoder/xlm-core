"""Extract plain model weights from Lightning module ``state_dict`` payloads."""

from __future__ import annotations

from typing import Any

import torch


def _maybe_strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }


def extract_model_only_from_lightning_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Strip ``model.`` / ``_orig_mod.`` prefixes from a LightningModule ``state_dict``.

    Keeps only tensor values suitable for ``safetensors`` / ``load_state_dict``.
    """
    sd = state_dict
    sd2 = _maybe_strip_prefix(sd, "model.")
    sd3 = _maybe_strip_prefix(sd2, "_orig_mod.")
    sd4 = _maybe_strip_prefix(sd3, "model._orig_mod.")

    for candidate in (sd4, sd3, sd2):
        if candidate is not sd and len(candidate) > 0:
            chosen = candidate
            break
    else:
        chosen = sd

    out: dict[str, torch.Tensor] = {}
    for k, v in chosen.items():
        if isinstance(v, torch.Tensor):
            out[k] = v
    return out


def tensor_state_dict_from_checkpoint_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Load a consolidated Lightning ``torch.save`` dict and return model-only tensors.

    Expects a top-level ``state_dict`` key (standard Lightning checkpoint after consolidation).
    """
    raw = checkpoint.get("state_dict")
    if not isinstance(raw, dict):
        raise ValueError(
            "Checkpoint must contain a 'state_dict' dict (consolidated Lightning format)."
        )
    return extract_model_only_from_lightning_state_dict(raw)
