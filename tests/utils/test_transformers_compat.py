"""RoPE init compatibility across transformers 4 and 5."""

import torch
from transformers.modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS as HF_ROPE_INIT_FUNCTIONS,
)
from transformers.modeling_rope_utils import (
    _compute_default_rope_parameters,
)

from xlm.utils.model_loading import (
    ROPE_INIT_FUNCTIONS,
    compute_default_rope_parameters,
    ensure_default_rope_init,
)


class _RopeConfig:
    """Dream-Coder / FlexMDM unscaled RoPE: theta 1e6, no scaling."""

    def __init__(self, rope_theta=1_000_000.0, rope_parameters=None):
        if rope_theta is not None:
            self.rope_theta = rope_theta
        self.rope_parameters = rope_parameters
        self.hidden_size = 4096
        self.num_attention_heads = 32


def test_exposes_transformers_map_when_default_present():
    assert "default" in HF_ROPE_INIT_FUNCTIONS
    assert ROPE_INIT_FUNCTIONS is HF_ROPE_INIT_FUNCTIONS
    assert (
        ROPE_INIT_FUNCTIONS["default"] is HF_ROPE_INIT_FUNCTIONS["default"]
    )


def test_default_matches_transformers4_unscaled_rope():
    config = _RopeConfig()
    inv_freq, scale = compute_default_rope_parameters(config)
    hf_inv, hf_scale = _compute_default_rope_parameters(config)
    torch.testing.assert_close(inv_freq, hf_inv)
    assert scale == 1.0
    assert hf_scale == 1.0
    # head_dim = 4096 / 32; pairs of rotary dims.
    assert inv_freq.shape == (64,)


def test_rope_parameters_theta_when_attribute_missing():
    config = _RopeConfig(
        rope_theta=None, rope_parameters={"rope_theta": 1_000_000.0}
    )
    via_attr = _RopeConfig(rope_theta=1_000_000.0)
    inv_freq, scale = compute_default_rope_parameters(config)
    expected, _ = compute_default_rope_parameters(via_attr)
    torch.testing.assert_close(inv_freq, expected)
    assert scale == 1.0


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
    inv_freq, scale = patched["default"](_RopeConfig())
    hf_inv, hf_scale = _compute_default_rope_parameters(_RopeConfig())
    torch.testing.assert_close(inv_freq, hf_inv)
    assert scale == hf_scale == 1.0


def test_ensure_default_returns_same_mapping_when_key_exists():
    sentinel = object()
    original = {"default": sentinel, "yarn": sentinel}
    assert ensure_default_rope_init(original) is original
