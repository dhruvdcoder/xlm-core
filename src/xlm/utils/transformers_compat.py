"""Shims for transformers APIs that differ between v4 and v5.

Sits next to ``xlm.utils.model_loading``, which picks
``no_init_weights`` from ``transformers.initialization`` (v5) or
``transformers.modeling_utils`` (v4).
"""

from typing import Any, Callable, Mapping, Optional, Tuple

import torch
from transformers.modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS as _HF_ROPE_INIT_FUNCTIONS,
)


def compute_default_rope_parameters(
    config: Optional[Any] = None,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs: Any,
) -> Tuple[torch.Tensor, float]:
    """Inverse frequencies for original (unscaled) RoPE.

    Matches ``transformers`` <=4 ``ROPE_INIT_FUNCTIONS["default"]``
    (``_compute_default_rope_parameters``): base ``config.rope_theta``
    (Dream-Coder / FlexMDM: ``1_000_000``) and attention scaling ``1.0``.
    On transformers 5, ``rope_theta`` may live in ``rope_parameters``.
    """
    del seq_len  # unused for unscaled RoPE; kept for the init-fn signature
    if config is not None:
        base = getattr(config, "rope_theta", None)
        rope_parameters = getattr(config, "rope_parameters", None)
        if base is None and isinstance(rope_parameters, dict):
            base = rope_parameters.get("rope_theta")
        if base is None:
            base = rope_kwargs.get("base", 10000.0)
        partial_rotary_factor = getattr(
            config, "partial_rotary_factor", 1.0
        )
        if partial_rotary_factor is None:
            partial_rotary_factor = 1.0
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
    else:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]

    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, 1.0


def ensure_default_rope_init(
    rope_init_functions: Mapping[str, Callable[..., Any]],
) -> Mapping[str, Callable[..., Any]]:
    """Return a RoPE init map that contains ``"default"``.

    If ``"default"`` is already present, the mapping is returned
    unchanged (transformers 4). Otherwise a shallow copy is returned
    with unscaled RoPE filled in. Yarn, linear, dynamic, and llama3
    entries are not replaced.
    """
    if "default" in rope_init_functions:
        return rope_init_functions
    patched = dict(rope_init_functions)
    patched["default"] = compute_default_rope_parameters
    return patched


ROPE_INIT_FUNCTIONS = ensure_default_rope_init(_HF_ROPE_INIT_FUNCTIONS)
