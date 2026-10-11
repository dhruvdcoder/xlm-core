"""Unit tests for Harness optimizer-coverage helpers (no distributed setup).

The FSDP-specific behaviour is covered by
tests/integration/test_fsdp_optimizer_coverage.py; here we check the plain
single-process semantics of Harness.optimizer_named_parameters and
Harness.check_optimizer_covers_trainable_params.
"""

from __future__ import annotations

import logging

import lightning as L
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from xlm.harness import Harness, _strip_wrapper_prefixes


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 8)


def _bare_harness(allow_partial: bool = False) -> Harness:
    harness = Harness.__new__(Harness)
    L.LightningModule.__init__(harness)
    harness.model = _TinyModel()
    harness.config = OmegaConf.create(
        {"lightning_module": {"allow_partial_optimizer_coverage": allow_partial}}
    )
    return harness


def test_fsdp_root_is_none_without_trainer():
    assert _bare_harness().fsdp_root() is None


def test_optimizer_named_parameters_matches_model_without_fsdp():
    harness = _bare_harness()
    names = [n for n, _ in harness.optimizer_named_parameters()]
    assert names == [n for n, _ in harness.model.named_parameters()]


def test_coverage_check_passes_when_all_trainable_params_present():
    harness = _bare_harness()
    opt = torch.optim.SGD(harness.model.parameters(), lr=0.1)
    harness.check_optimizer_covers_trainable_params(opt)


def test_coverage_check_ignores_frozen_params():
    harness = _bare_harness()
    harness.model.embed.weight.requires_grad_(False)
    opt = torch.optim.SGD(
        [p for p in harness.model.parameters() if p.requires_grad], lr=0.1
    )
    harness.check_optimizer_covers_trainable_params(opt)


def test_coverage_check_raises_and_names_missing_params():
    harness = _bare_harness()
    opt = torch.optim.SGD(harness.model.head.parameters(), lr=0.1)
    with pytest.raises(RuntimeError) as excinfo:
        harness.check_optimizer_covers_trainable_params(opt)
    message = str(excinfo.value)
    assert "embed.weight" in message
    assert "norm.weight" in message
    assert "head.weight" not in message
    assert "allow_partial_optimizer_coverage" in message


def test_coverage_check_opt_out_downgrades_to_warning(caplog):
    harness = _bare_harness(allow_partial=True)
    opt = torch.optim.SGD(harness.model.head.parameters(), lr=0.1)
    with caplog.at_level(logging.WARNING, logger="xlm.harness"):
        harness.check_optimizer_covers_trainable_params(opt)
    assert any("embed.weight" in rec.getMessage() for rec in caplog.records)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("_fsdp_wrapped_module._flat_param", "_flat_param"),
        (
            "_fsdp_wrapped_module.model.layers.0._fsdp_wrapped_module.mlp.w",
            "layers.0.mlp.w",
        ),
        (
            "_fsdp_wrapped_module.model.layers.1._fsdp_wrapped_module."
            "_checkpoint_wrapped_module.attn.q",
            "layers.1.attn.q",
        ),
        ("model.embed.weight", "embed.weight"),
        ("embed.weight", "embed.weight"),
        ("_orig_mod.model.head.weight", "head.weight"),
    ],
)
def test_strip_wrapper_prefixes(raw: str, expected: str):
    assert _strip_wrapper_prefixes(raw) == expected
