"""Unit tests for :mod:`xlm.utils.batch_size`."""

from unittest.mock import MagicMock

import pytest
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy

from xlm.utils.batch_size import determine_per_device_batch_size, is_ddp


def _trainer_with_strategy(strategy):
    t = MagicMock()
    t.strategy = strategy
    return t


class TestIsDdp:
    def test_returns_false_when_trainer_is_none(self):
        assert is_ddp(None) is False

    def test_returns_true_for_ddp_strategy(self):
        # DDPStrategy.__init__ calls into Lightning internals, but isinstance
        # only needs an instance with the right type, so use a Mock with spec.
        ddp_like = MagicMock(spec=DDPStrategy)
        assert is_ddp(_trainer_with_strategy(ddp_like)) is True

    def test_returns_false_for_single_device_strategy(self):
        single_like = MagicMock(spec=SingleDeviceStrategy)
        assert is_ddp(_trainer_with_strategy(single_like)) is False

    def test_unknown_strategy_raises(self):
        class _Other:
            pass

        with pytest.raises(ValueError, match="strategy"):
            is_ddp(_trainer_with_strategy(_Other()))


class TestDeterminePerDeviceBatchSize:
    def test_raises_when_trainer_is_none(self):
        with pytest.raises(ValueError, match="Trainer is not setup"):
            determine_per_device_batch_size(None, batch_size=8)

    def test_non_ddp_returns_input_batch_size(self):
        single_like = MagicMock(spec=SingleDeviceStrategy)
        trainer = _trainer_with_strategy(single_like)
        assert determine_per_device_batch_size(trainer, batch_size=8) == 8

    def test_ddp_divides_by_devices_nodes_accum(self):
        ddp_like = MagicMock(spec=DDPStrategy)
        trainer = _trainer_with_strategy(ddp_like)
        trainer.num_nodes = 2
        trainer.num_devices = 4
        trainer.accumulate_grad_batches = 2
        # 64 / (2 * 4 * 2) = 4
        assert determine_per_device_batch_size(trainer, batch_size=64) == 4

    def test_ddp_indivisible_batch_size_raises(self):
        ddp_like = MagicMock(spec=DDPStrategy)
        trainer = _trainer_with_strategy(ddp_like)
        trainer.num_nodes = 2
        trainer.num_devices = 4
        trainer.accumulate_grad_batches = 1
        with pytest.raises(ValueError, match="not divisible"):
            determine_per_device_batch_size(trainer, batch_size=10)
