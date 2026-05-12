"""Unit tests for :class:`xlm.utils.ema.EMACallback`."""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from xlm.utils.ema import EMACallback


class _TinyModule(nn.Module):
    """A bare-bones stand-in for a Lightning module."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


@pytest.fixture()
def tiny_module():
    return _TinyModule()


@pytest.fixture()
def trainer_mock():
    t = MagicMock()
    t.accumulate_grad_batches = 2
    return t


class TestEMACallbackConstruction:
    def test_stores_init_args(self):
        cb = EMACallback(decay=0.9, use_num_updates=False, apply_ema_at_train_end=False)
        assert cb.decay == 0.9
        assert cb.use_num_updates is False
        assert cb.apply_ema_at_train_end is False
        assert cb.ema is None

    def test_default_kwargs(self):
        cb = EMACallback(decay=0.5)
        assert cb.use_num_updates is True
        assert cb.apply_ema_at_train_end is True


class TestEMACallbackLifecycle:
    def test_on_train_start_creates_ema(self, tiny_module, trainer_mock):
        cb = EMACallback(decay=0.9)
        assert cb.ema is None
        cb.on_train_start(trainer_mock, tiny_module)
        assert cb.ema is not None

    def test_on_train_start_is_idempotent(self, tiny_module, trainer_mock):
        cb = EMACallback(decay=0.9)
        cb.on_train_start(trainer_mock, tiny_module)
        first = cb.ema
        cb.on_train_start(trainer_mock, tiny_module)
        # Second call must not re-create the EMA object (so resumed checkpoints
        # keep their stats).
        assert cb.ema is first

    def test_on_train_batch_end_updates_only_on_full_step(
        self, tiny_module, trainer_mock
    ):
        cb = EMACallback(decay=0.9)
        cb.on_train_start(trainer_mock, tiny_module)

        cb.ema = MagicMock(wraps=cb.ema)
        # accumulate_grad_batches = 2 -> updates only when (batch_idx+1) % 2 == 0
        cb.on_train_batch_end(trainer_mock, tiny_module, None, None, batch_idx=0)
        cb.on_train_batch_end(trainer_mock, tiny_module, None, None, batch_idx=1)
        cb.on_train_batch_end(trainer_mock, tiny_module, None, None, batch_idx=2)
        cb.on_train_batch_end(trainer_mock, tiny_module, None, None, batch_idx=3)
        assert cb.ema.update.call_count == 2

    def test_validation_hooks_are_noop_when_ema_is_none(
        self, tiny_module, trainer_mock
    ):
        cb = EMACallback(decay=0.9)
        # No on_train_start has been invoked: cb.ema is None and the hooks
        # must short-circuit.
        cb.on_validation_start(trainer_mock, tiny_module)
        cb.on_validation_end(trainer_mock, tiny_module)
        cb.on_test_start(trainer_mock, tiny_module)
        cb.on_test_end(trainer_mock, tiny_module)
        cb.on_predict_start(trainer_mock, tiny_module)
        cb.on_predict_end(trainer_mock, tiny_module)

    def test_validation_hooks_invoke_store_copy_restore(
        self, tiny_module, trainer_mock
    ):
        cb = EMACallback(decay=0.9)
        cb.on_train_start(trainer_mock, tiny_module)
        cb.ema = MagicMock(wraps=cb.ema)

        cb.on_validation_start(trainer_mock, tiny_module)
        cb.ema.store.assert_called_once()
        cb.ema.copy_to.assert_called_once()

        cb.on_validation_end(trainer_mock, tiny_module)
        cb.ema.restore.assert_called_once()


class TestEMACallbackCheckpoint:
    def test_save_checkpoint_writes_state_dict(self, tiny_module, trainer_mock):
        cb = EMACallback(decay=0.9)
        cb.on_train_start(trainer_mock, tiny_module)
        ckpt: dict = {}
        cb.on_save_checkpoint(trainer_mock, tiny_module, ckpt)
        assert "ema" in ckpt

    def test_save_checkpoint_noop_when_ema_is_none(
        self, tiny_module, trainer_mock
    ):
        cb = EMACallback(decay=0.9)
        ckpt: dict = {}
        cb.on_save_checkpoint(trainer_mock, tiny_module, ckpt)
        assert ckpt == {}

    def test_save_load_round_trip(self, tiny_module, trainer_mock):
        cb_save = EMACallback(decay=0.9)
        cb_save.on_train_start(trainer_mock, tiny_module)
        ckpt: dict = {}
        cb_save.on_save_checkpoint(trainer_mock, tiny_module, ckpt)

        cb_load = EMACallback(decay=0.9)
        cb_load.on_load_checkpoint(trainer_mock, tiny_module, ckpt)
        assert cb_load.ema is not None
