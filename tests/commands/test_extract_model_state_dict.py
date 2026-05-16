"""Unit tests for :mod:`xlm.commands.extract_model_state_dict`."""

import logging

import pytest
import torch

from xlm.commands.extract_model_state_dict import (
    _step_info_prefix,
    filter_model_state_dict,
    print_checkpoint_info,
)


class TestPrintCheckpointInfo:
    def test_logs_scalar_keys(self, caplog):
        caplog.set_level(
            logging.INFO, logger="xlm.commands.extract_model_state_dict"
        )
        full = {
            "epoch": 3,
            "global_step": 1000,
            "name": "run",
            "trained": True,
            "loss": 0.5,
            # Tensors should be skipped (only scalars are logged per-key).
            "state_dict": torch.zeros(1),
        }
        print_checkpoint_info(full)
        log = "\n".join(r.message for r in caplog.records)
        assert "Checkpoint keys" in log
        assert "epoch" in log and "1000" in log
        assert "0.5" in log
        assert "True" in log


class TestStepInfoPrefix:
    def test_includes_epoch_and_step(self):
        full = {"epoch": 2, "global_step": 1000}
        prefix = _step_info_prefix(full)
        assert "epoch=2" in prefix
        assert "step=1000" in prefix
        assert prefix.endswith("_")

    def test_missing_keys_become_none(self):
        prefix = _step_info_prefix({})
        assert "epoch=None" in prefix
        assert "step=None" in prefix


class TestFilterModelStateDict:
    def test_keeps_only_model_keys_and_strips_prefix(self):
        full = {
            "model.encoder.weight": torch.tensor([1.0]),
            "model.encoder.bias": torch.tensor([2.0]),
            "ema.weight": torch.tensor([3.0]),
            "loss.scale": torch.tensor([4.0]),
            "global_step": 7,
        }
        out = filter_model_state_dict(full)
        assert set(out.keys()) == {"encoder.weight", "encoder.bias"}
        assert torch.equal(out["encoder.weight"], torch.tensor([1.0]))

    def test_empty_when_no_model_keys(self):
        out = filter_model_state_dict({"ema.x": torch.tensor([1.0])})
        assert out == {}

    def test_handles_nested_model_prefix(self):
        full = {"model.model.deep.weight": torch.tensor([1.0])}
        out = filter_model_state_dict(full)
        assert list(out.keys()) == ["model.deep.weight"]
