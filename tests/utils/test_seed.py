"""Unit tests for :mod:`xlm.utils.seed`."""

import random

import numpy as np
import pytest
import torch

import xlm.utils.seed as seed_mod
from xlm.utils.seed import (
    _collect_rng_states,
    _set_rng_states,
    seed_everything,
    seed_everything_simple,
)


@pytest.fixture()
def patched_get_rank(monkeypatch):
    """Inject a stub ``_get_rank`` into ``xlm.utils.seed``.

    The module references ``_get_rank()`` at module level inside
    :func:`seed_everything` but never imports/defines it, so calls would
    otherwise raise ``NameError``.
    """
    monkeypatch.setattr(
        seed_mod, "_get_rank", lambda: 0, raising=False
    )


@pytest.fixture(autouse=True)
def _restore_seed_env(monkeypatch):
    # Snapshot env vars touched by seed_everything; let monkeypatch undo them.
    for key in ("HYDRA_GLOBAL_SEED", "HYDRA_SEED_WORKERS"):
        monkeypatch.delenv(key, raising=False)
    yield


class TestSeedEverythingSimple:
    def test_torch_reproducibility(self):
        seed_everything_simple(42)
        a = torch.rand(4)
        seed_everything_simple(42)
        b = torch.rand(4)
        assert torch.equal(a, b)

    def test_numpy_reproducibility(self):
        seed_everything_simple(123)
        a = np.random.rand(4)
        seed_everything_simple(123)
        b = np.random.rand(4)
        assert np.allclose(a, b)

    def test_python_reproducibility(self):
        seed_everything_simple(7)
        a = [random.random() for _ in range(3)]
        seed_everything_simple(7)
        b = [random.random() for _ in range(3)]
        assert a == b


class TestSeedEverything:
    def test_returns_seed_and_sets_env(self, patched_get_rank):
        out = seed_everything(42)
        assert out == 42
        import os

        assert os.environ["HYDRA_GLOBAL_SEED"] == "42"
        assert os.environ["HYDRA_SEED_WORKERS"] == "0"

    def test_workers_flag_sets_env(self, patched_get_rank):
        seed_everything(7, workers=True)
        import os

        assert os.environ["HYDRA_SEED_WORKERS"] == "1"

    def test_reads_seed_from_env_when_called_with_none(
        self, patched_get_rank, monkeypatch
    ):
        monkeypatch.setenv("HYDRA_GLOBAL_SEED", "11")
        out = seed_everything(None)
        assert out == 11

    def test_defaults_to_zero_when_no_seed_and_no_env(
        self, patched_get_rank
    ):
        out = seed_everything(None)
        assert out == 0

    def test_invalid_env_seed_falls_back_to_zero(
        self, patched_get_rank, monkeypatch
    ):
        monkeypatch.setenv("HYDRA_GLOBAL_SEED", "not_an_int")
        out = seed_everything(None)
        assert out == 0

    def test_out_of_range_seed_clamps_to_zero(self, patched_get_rank):
        # uint32 max is 2**32 - 1
        out = seed_everything(2**32 + 5)
        assert out == 0

    def test_non_int_seed_is_coerced(self, patched_get_rank):
        out = seed_everything(42.0)
        assert out == 42

    def test_seeds_are_actually_applied(self, patched_get_rank):
        seed_everything(2024)
        a = torch.rand(3)
        seed_everything(2024)
        b = torch.rand(3)
        assert torch.equal(a, b)


class TestRngStates:
    def test_round_trip_preserves_state(self):
        # Snapshot, then perturb, then restore -> next sample matches snapshot.
        states = _collect_rng_states(include_cuda=False)
        # Disturb
        torch.rand(5)
        np.random.rand(5)
        random.random()
        # Restore
        _set_rng_states({k: v for k, v in states.items() if k != "torch.cuda"})
        a_torch = torch.rand(3)
        a_np = np.random.rand(3)
        a_py = random.random()
        # And again from the same snapshot
        _set_rng_states({k: v for k, v in states.items() if k != "torch.cuda"})
        b_torch = torch.rand(3)
        b_np = np.random.rand(3)
        b_py = random.random()
        assert torch.equal(a_torch, b_torch)
        assert np.allclose(a_np, b_np)
        assert a_py == b_py

    def test_collect_includes_torch_cuda_key(self):
        states = _collect_rng_states(include_cuda=True)
        assert "torch.cuda" in states

    def test_collect_skips_cuda_when_disabled(self):
        states = _collect_rng_states(include_cuda=False)
        assert "torch.cuda" not in states
