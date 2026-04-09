from xlm.utils.fsdp_grouping import build_fsdp_grouping_plan_from_config


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
