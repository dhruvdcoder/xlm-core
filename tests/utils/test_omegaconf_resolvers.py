"""Unit tests for :mod:`xlm.utils.omegaconf_resolvers`."""

import pytest
from omegaconf import DictConfig, OmegaConf

from xlm.utils.omegaconf_resolvers import (
    determine_accumulate_grad_batches,
    dictconfig_filter_key,
    register_resolvers,
    remove_keys_with_double_underscores,
)


_RESOLVER_NAMES = (
    "cwd",
    "device_count",
    "eval",
    "div_up",
    "num_cpus",
    "replace_str",
    "dict_to_list",
    "find_grad_accum",
    "if_else",
    "min",
    "max",
    "min_int",
    "max_int",
)


@pytest.fixture()
def registered_resolvers():
    """Register the project resolvers for the duration of a test."""
    for name in _RESOLVER_NAMES:
        OmegaConf.clear_resolver(name)
    register_resolvers()
    yield
    for name in _RESOLVER_NAMES:
        OmegaConf.clear_resolver(name)


class TestDetermineAccumulateGradBatches:
    def test_simple_divisible(self):
        assert (
            determine_accumulate_grad_batches(
                global_batch_size=64,
                per_device_batch_size=4,
                num_devices=2,
                num_nodes=1,
            )
            == 8
        )

    def test_cpu_branch_when_num_devices_is_zero(self):
        # ``num_devices=0`` is treated as a single CPU device.
        assert (
            determine_accumulate_grad_batches(
                global_batch_size=16,
                per_device_batch_size=4,
                num_devices=0,
                num_nodes=1,
            )
            == 4
        )

    def test_multi_node(self):
        assert (
            determine_accumulate_grad_batches(
                global_batch_size=128,
                per_device_batch_size=4,
                num_devices=4,
                num_nodes=2,
            )
            == 4
        )

    def test_non_divisible_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            determine_accumulate_grad_batches(
                global_batch_size=10,
                per_device_batch_size=4,
                num_devices=1,
                num_nodes=1,
            )


class TestDictConfigFilterKey:
    def test_keeps_only_matching_keys(self):
        cfg = OmegaConf.create({"a": 1, "b": 2, "c": 3})
        out = dictconfig_filter_key(cfg, lambda k: k != "b")
        assert dict(out) == {"a": 1, "c": 3}

    def test_recurses_into_nested(self):
        cfg = OmegaConf.create({"a": 1, "nested": {"x": 1, "y": 2}})
        out = dictconfig_filter_key(cfg, lambda k: k != "x")
        assert OmegaConf.to_container(out) == {"a": 1, "nested": {"y": 2}}


class TestRemoveKeysWithDoubleUnderscores:
    def test_drops_dunder_keys(self):
        cfg = OmegaConf.create(
            {"a": 1, "__internal": 2, "nested": {"b": 3, "__hidden": 4}}
        )
        out = remove_keys_with_double_underscores(cfg)
        assert OmegaConf.to_container(out) == {
            "a": 1,
            "nested": {"b": 3},
        }


class TestRegisterResolvers:
    def test_eval_resolver(self, registered_resolvers):
        cfg = OmegaConf.create({"x": "${eval:2 + 3}"})
        assert cfg.x == 5

    def test_div_up_resolver(self, registered_resolvers):
        cfg = OmegaConf.create({"x": "${div_up:7,3}"})
        assert cfg.x == 3

    def test_if_else_resolver(self, registered_resolvers):
        cfg = OmegaConf.create(
            {
                "t": "${if_else:1,a,b}",
                "f": "${if_else:0,a,b}",
            }
        )
        assert cfg.t == "a"
        assert cfg.f == "b"

    def test_min_max_resolvers(self, registered_resolvers):
        cfg = OmegaConf.create(
            {
                "lo": "${min:1.5,2,3}",
                "hi": "${max:1,2.5,3}",
                "lo_i": "${min_int:1,2,3}",
                "hi_i": "${max_int:1,2,3}",
            }
        )
        assert cfg.lo == 1.5
        assert cfg.hi == 3.0
        assert cfg.lo_i == 1
        assert cfg.hi_i == 3

    def test_replace_str_resolver(self, registered_resolvers):
        cfg = OmegaConf.create({"x": "${replace_str:hello,l,L}"})
        assert cfg.x == "heLLo"

    def test_find_grad_accum_resolver(self, registered_resolvers):
        cfg = OmegaConf.create({"x": "${find_grad_accum:32,4,2,1}"})
        assert cfg.x == 4

    def test_num_cpus_returns_positive_int(self, registered_resolvers):
        cfg = OmegaConf.create({"x": "${num_cpus:}"})
        assert isinstance(cfg.x, int) and cfg.x > 0

    def test_cwd_resolver_returns_string(self, registered_resolvers):
        cfg = OmegaConf.create({"x": "${cwd:}"})
        assert isinstance(cfg.x, str) and cfg.x
