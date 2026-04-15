import pytest
import torch

from xlm.backbones.dream.modeling_dream import DreamDecoderLayer
from xlm.utils.fsdp_grouping import (
    build_fsdp_grouping_plan_from_config,
    fsdp_bf16_mixed_precision,
    make_layer_wrap_policy,
)


def test_make_layer_wrap_policy_resolves_dream_decoder_layer():
    classes = make_layer_wrap_policy(
        "xlm.backbones.dream.modeling_dream.DreamDecoderLayer"
    )
    assert classes == {DreamDecoderLayer}


def test_fsdp_bf16_mixed_precision_dtypes():
    mp = fsdp_bf16_mixed_precision()
    assert mp.param_dtype == torch.bfloat16
    assert mp.reduce_dtype == torch.float32
    assert mp.buffer_dtype == torch.float32


def test_fsdp_strategy_accepts_wrap_policy_and_mixed_precision():
    pytest.importorskip("lightning")
    from lightning.pytorch.strategies import FSDPStrategy

    p = make_layer_wrap_policy(
        "xlm.backbones.dream.modeling_dream.DreamDecoderLayer"
    )
    mp = fsdp_bf16_mixed_precision()
    strategy = FSDPStrategy(
        auto_wrap_policy=p,
        mixed_precision=mp,
        sharding_strategy="FULL_SHARD",
        cpu_offload=False,
        use_orig_params=False,
    )
    assert strategy is not None


def test_build_fsdp_grouping_plan_from_config_plain_dict():
    cfg = {
        "embed": {"path": "model.embed_tokens", "wrap": False},
        "layers": {
            "path_template": "model.layers.{i}",
            "wrap": False,
            "count": 3,
        },
        "head": {"path": "lm_head", "wrap": True},
    }
    plan = build_fsdp_grouping_plan_from_config(cfg, context={})
    assert plan == [
        ("model.embed_tokens", False),
        ("model.layers.0", False),
        ("model.layers.1", False),
        ("model.layers.2", False),
        ("lm_head", True),
    ]


def test_flops_helpers_import():
    from xlm.utils.flops_estimation import (
        attention_flops_per_token,
        get_num_flop_per_token,
    )

    assert attention_flops_per_token(1, 128, 4096, False) > 0
    assert get_num_flop_per_token(1_000_000, 32, 4096, 128) > 0
