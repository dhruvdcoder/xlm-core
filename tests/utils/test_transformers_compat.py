"""RoPE init compatibility across transformers 4 and 5."""

import torch
from transformers import modeling_rope_utils
from transformers.modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS as HF_ROPE_INIT_FUNCTIONS,
)

from xlm.utils.model_loading import (
    ROPE_INIT_FUNCTIONS,
    compute_default_rope_parameters,
    ensure_default_rope_init,
)


def _hf_default_rope_parameters():
    """Private transformers<=4 helper. Absent on transformers 5."""
    return getattr(
        modeling_rope_utils, "_compute_default_rope_parameters", None
    )


def _unscaled_rope(rope_theta: float, dim: int):
    """Original RoPE inverse frequencies, attention scale 1.0."""
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float)
            / dim
        )
    )
    return inv_freq, 1.0


class _RopeConfig:
    """Dream-Coder / FlexMDM unscaled RoPE: theta 1e6, no scaling."""

    def __init__(self, rope_theta=1_000_000.0, rope_parameters=None):
        if rope_theta is not None:
            self.rope_theta = rope_theta
        self.rope_parameters = rope_parameters
        self.hidden_size = 4096
        self.num_attention_heads = 32


def test_exposes_transformers_map_when_default_present():
    assert "default" in ROPE_INIT_FUNCTIONS
    if "default" in HF_ROPE_INIT_FUNCTIONS:
        assert ROPE_INIT_FUNCTIONS is HF_ROPE_INIT_FUNCTIONS
        assert (
            ROPE_INIT_FUNCTIONS["default"]
            is HF_ROPE_INIT_FUNCTIONS["default"]
        )
    else:
        # transformers 5 drops the key; the shim fills unscaled RoPE.
        assert ROPE_INIT_FUNCTIONS is not HF_ROPE_INIT_FUNCTIONS
        assert (
            ROPE_INIT_FUNCTIONS["default"] is compute_default_rope_parameters
        )
        for key, fn in HF_ROPE_INIT_FUNCTIONS.items():
            assert ROPE_INIT_FUNCTIONS[key] is fn


def test_default_matches_transformers4_unscaled_rope():
    config = _RopeConfig()
    inv_freq, scale = compute_default_rope_parameters(config)
    expected, expected_scale = _unscaled_rope(config.rope_theta, dim=128)
    torch.testing.assert_close(inv_freq, expected)
    assert scale == expected_scale == 1.0
    # head_dim = 4096 / 32; pairs of rotary dims.
    assert inv_freq.shape == (64,)

    hf_default = _hf_default_rope_parameters()
    if hf_default is not None:
        hf_inv, hf_scale = hf_default(config)
        torch.testing.assert_close(inv_freq, hf_inv)
        assert hf_scale == 1.0


def test_rope_parameters_theta_when_attribute_missing():
    config = _RopeConfig(
        rope_theta=None, rope_parameters={"rope_theta": 1_000_000.0}
    )
    via_attr = _RopeConfig(rope_theta=1_000_000.0)
    inv_freq, scale = compute_default_rope_parameters(config)
    expected, _ = compute_default_rope_parameters(via_attr)
    torch.testing.assert_close(inv_freq, expected)
    assert scale == 1.0
    unscaled, _ = _unscaled_rope(1_000_000.0, dim=128)
    torch.testing.assert_close(inv_freq, unscaled)


def test_fills_default_without_touching_other_rope_types():
    """Transformers 5 drops ``"default"`` and keeps the scaled types."""
    without_default = {
        key: fn
        for key, fn in HF_ROPE_INIT_FUNCTIONS.items()
        if key != "default"
    }
    assert "yarn" in without_default
    patched = ensure_default_rope_init(without_default)
    assert patched is not without_default
    assert "default" not in without_default
    assert patched["default"] is compute_default_rope_parameters
    for key, fn in without_default.items():
        assert patched[key] is fn
    config = _RopeConfig()
    inv_freq, scale = patched["default"](config)
    expected, expected_scale = _unscaled_rope(config.rope_theta, dim=128)
    torch.testing.assert_close(inv_freq, expected)
    assert scale == expected_scale == 1.0

    hf_default = _hf_default_rope_parameters()
    if hf_default is not None:
        hf_inv, hf_scale = hf_default(config)
        torch.testing.assert_close(inv_freq, hf_inv)
        assert hf_scale == 1.0


def test_ensure_default_returns_same_mapping_when_key_exists():
    sentinel = object()
    original = {"default": sentinel, "yarn": sentinel}
    assert ensure_default_rope_init(original) is original
