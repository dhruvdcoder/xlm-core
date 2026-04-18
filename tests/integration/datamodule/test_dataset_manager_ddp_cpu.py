"""CPU multi-process DDP integration tests for ``DatasetManager``.

These tests exercise the genuinely distributed code paths --
``split_dataset_by_node``, ``StatefulDistributedSampler``,
``set_epoch`` reshuffling and the ``num_shards / world_size /
num_workers`` validation in
:meth:`xlm.datamodule.DatasetManager.get_dataloader` -- in real
multi-process Python jobs launched via ``torch.distributed.run`` with
the ``gloo`` backend.

No GPU is required.  All datasets are in-memory (see
:mod:`tests.datamodule_helpers`); the subprocess entrypoint is at
:mod:`tests.integration._scripts.ddp_dsm_entrypoint`.

Each test:

1. Builds a JSON-serialisable config describing how ``DatasetManager``
   should be constructed and iterated.
2. Calls :func:`tests.integration._runner.run_cpu_distributed` which
   spawns ``world_size`` python processes, waits for them, and returns
   one parsed result dict per rank.
3. Asserts on per-rank coverage / overlap / shapes.

These tests are slow (~10-20 s each) because they fork Python and
initialise ``torch.distributed``, so they are gated by both the
``integration`` and ``ddp`` markers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.datamodule_helpers import EXAMPLE_TOKEN_LEN
from tests.integration._runner import run_cpu_distributed


pytestmark = [pytest.mark.integration, pytest.mark.ddp]


ENTRYPOINT = "tests.integration._scripts.ddp_dsm_entrypoint"

# Sizes of the in-memory datasets defined in tests.datamodule_helpers.
TRAIN_SIZE = 17
LARGE_SIZE = 60


def _all_ids(result: Dict[str, Any], epoch: int = 0) -> List[int]:
    return list(result["epochs"][epoch]["ids"])


# ---------------------------------------------------------------------------
# Iterable + DDP: split_dataset_by_node coverage / non-overlap
# ---------------------------------------------------------------------------


class TestIterableDdpSplitByNode:
    """Iterable-dataset DDP path (case 1 in get_dataloader)."""

    def test_world_size_2_iterable_shards_4_full_coverage(self, result_dir: Path):
        """Two ranks with 4 shards must jointly cover every example exactly once."""
        config = {
            "dsm_kwargs": {
                "full_name": "mem/raw_large/train",
                "iterable_dataset_shards": 4,
                "dataloader_kwargs": {
                    "batch_size": 4,
                    "num_workers": 1,
                    "pin_memory": False,
                },
            },
            "run": {"num_epochs": 1, "set_epoch_each_epoch": False},
        }
        results = run_cpu_distributed(
            script_module=ENTRYPOINT,
            world_size=2,
            result_dir=result_dir,
            config=config,
        )
        assert len(results) == 2
        rank0_ids = _all_ids(results[0])
        rank1_ids = _all_ids(results[1])
        # Non-overlap.
        assert set(rank0_ids).isdisjoint(set(rank1_ids))
        # Joint coverage of the entire dataset (id range 1000 .. 1059).
        assert sorted(rank0_ids + rank1_ids) == list(range(1000, 1000 + LARGE_SIZE))
        # Iterable path is hit, not the map-style sampler path.
        assert all(r["is_iterable_dataset"] for r in results)

    def test_world_size_4_iterable_shards_4_full_coverage(self, result_dir: Path):
        """Stress-test with one shard per rank."""
        config = {
            "dsm_kwargs": {
                "full_name": "mem/raw_large/train",
                "iterable_dataset_shards": 4,
                "dataloader_kwargs": {
                    "batch_size": 4,
                    "num_workers": 1,
                    "pin_memory": False,
                },
            },
            "run": {"num_epochs": 1, "set_epoch_each_epoch": False},
        }
        results = run_cpu_distributed(
            script_module=ENTRYPOINT,
            world_size=4,
            result_dir=result_dir,
            config=config,
        )
        assert len(results) == 4
        seen: List[int] = []
        for i in range(4):
            seen.extend(_all_ids(results[i]))
            for j in range(i + 1, 4):
                assert set(_all_ids(results[i])).isdisjoint(
                    set(_all_ids(results[j]))
                ), f"ranks {i} and {j} overlap"
        assert sorted(seen) == list(range(1000, 1000 + LARGE_SIZE))


# ---------------------------------------------------------------------------
# Iterable + DDP: num_shards / num_workers / world_size validation
# ---------------------------------------------------------------------------


class TestIterableDdpShardValidation:
    """get_dataloader case 1 raises if num_shards_per_worker > batch_size."""

    def test_too_many_shards_subprocess_reports_error(self, result_dir: Path):
        # num_shards=8, world_size=2, num_workers=1 -> per-worker shards = 4.
        # batch_size=2 < 4 should trigger ValueError(num_shards_per_worker)
        # inside the subprocess -- bubbled up via ok=False / error.
        config = {
            "dsm_kwargs": {
                "full_name": "mem/raw_large/train",
                "iterable_dataset_shards": 8,
                "dataloader_kwargs": {
                    "batch_size": 2,
                    "num_workers": 1,
                    "pin_memory": False,
                },
            },
            "run": {"num_epochs": 1, "set_epoch_each_epoch": False},
        }
        with pytest.raises(AssertionError, match="num_shards_per_worker"):
            run_cpu_distributed(
                script_module=ENTRYPOINT,
                world_size=2,
                result_dir=result_dir,
                config=config,
            )


# ---------------------------------------------------------------------------
# Iterable + DDP: set_epoch reshuffling
# ---------------------------------------------------------------------------


class TestIterableDdpSetEpoch:
    """``dsm.set_epoch`` must change the per-rank ordering between epochs.

    ``DatasetManager.setup`` applies ``shuffle(buffer_size, seed)`` *before*
    ``split_dataset_by_node``, so a non-zero shuffle buffer crosses shard
    boundaries: the *content* assigned to each rank is allowed to change
    between epochs.  The invariants we care about are therefore:

    * Within any single epoch, the union of ranks' ``ids`` covers every
      example exactly once (no examples lost / duplicated).
    * Between epochs, at least one rank observes a different ordering --
      otherwise ``set_epoch`` is being silently dropped.
    """

    def test_shuffle_buffer_set_epoch_changes_order(self, result_dir: Path):
        config = {
            "dsm_kwargs": {
                "full_name": "mem/raw_large/train",
                "iterable_dataset_shards": 4,
                "shuffle_buffer_size": 16,
                "shuffle_seed": 7,
                "dataloader_kwargs": {
                    "batch_size": 4,
                    "num_workers": 1,
                    "pin_memory": False,
                },
            },
            "run": {"num_epochs": 2, "set_epoch_each_epoch": True},
        }
        results = run_cpu_distributed(
            script_module=ENTRYPOINT,
            world_size=2,
            result_dir=result_dir,
            config=config,
        )
        # Per-epoch joint coverage of all ranks must be the full dataset.
        for epoch in (0, 1):
            joint = sorted(
                i for r in results for i in r["epochs"][epoch]["ids"]
            )
            assert joint == list(range(1000, 1000 + LARGE_SIZE)), (
                f"epoch {epoch} joint coverage incomplete: {joint}"
            )
        # At least one rank must observe a different ordering across epochs.
        differs = any(
            list(r["epochs"][0]["ids"]) != list(r["epochs"][1]["ids"])
            for r in results
        )
        assert differs, "set_epoch did not change any rank's ordering"


# ---------------------------------------------------------------------------
# Iterable + DDP: make_infinite -> bounded by max_batches_per_epoch
# ---------------------------------------------------------------------------


class TestIterableDdpMakeInfinite:
    """``make_infinite=True`` must keep producing batches indefinitely."""

    def test_make_infinite_repeats_within_one_epoch(self, result_dir: Path):
        # 60 examples, world_size=2 -> 30 per rank per natural epoch.
        # batch_size=4, num_workers=1: 8 batches consumes 32 examples
        # which exceeds one natural epoch -- only possible if make_infinite
        # is honoured.
        config = {
            "dsm_kwargs": {
                "full_name": "mem/raw_large/train",
                "iterable_dataset_shards": 4,
                "make_infinite": True,
                "dataloader_kwargs": {
                    "batch_size": 4,
                    "num_workers": 1,
                    "pin_memory": False,
                },
            },
            "run": {
                "num_epochs": 1,
                "set_epoch_each_epoch": False,
                "max_batches_per_epoch": 12,
            },
        }
        results = run_cpu_distributed(
            script_module=ENTRYPOINT,
            world_size=2,
            result_dir=result_dir,
            config=config,
        )
        for r in results:
            ids = _all_ids(r)
            # 12 batches * 4 = 48 ids per rank -- impossible without infinity
            # because each rank only gets 30 unique ids per natural epoch.
            assert len(ids) == 48, (
                f"rank {r['rank']}: expected 48 ids with make_infinite=True, "
                f"got {len(ids)}"
            )


# ---------------------------------------------------------------------------
# Map-style + DDP: case 2 -- StatefulDistributedSampler
# ---------------------------------------------------------------------------


class TestMapStyleDdp:
    """Map-style DDP path (case 2 in get_dataloader)."""

    def test_world_size_2_map_style_full_coverage(self, result_dir: Path):
        # 17 examples / world=2 -- DistributedSampler default pads with
        # wrap-around so each rank sees ceil(17/2)=9 examples.
        config = {
            "dsm_kwargs": {
                "full_name": "mem/raw/train",
                # No iterable_dataset_shards -> map-style.
                "dataloader_kwargs": {
                    "batch_size": 3,
                    "num_workers": 0,
                    "pin_memory": False,
                },
            },
            "run": {"num_epochs": 1, "set_epoch_each_epoch": False},
        }
        results = run_cpu_distributed(
            script_module=ENTRYPOINT,
            world_size=2,
            result_dir=result_dir,
            config=config,
        )
        assert len(results) == 2
        for r in results:
            assert r["is_iterable_dataset"] is False
        rank0 = _all_ids(results[0])
        rank1 = _all_ids(results[1])
        # Each rank receives ceil(N/world_size) examples (DistributedSampler
        # default behaviour), and the union covers every original id.
        assert len(rank0) == len(rank1)
        union = set(rank0) | set(rank1)
        assert union == set(range(TRAIN_SIZE))

    def test_map_style_set_epoch_changes_per_rank_order(self, result_dir: Path):
        config = {
            "dsm_kwargs": {
                "full_name": "mem/raw/train",
                "dataloader_kwargs": {
                    "batch_size": 3,
                    "num_workers": 0,
                    "pin_memory": False,
                },
            },
            "run": {"num_epochs": 2, "set_epoch_each_epoch": True},
        }
        results = run_cpu_distributed(
            script_module=ENTRYPOINT,
            world_size=2,
            result_dir=result_dir,
            config=config,
        )
        # At least one rank should observe a different order across epochs --
        # StatefulDistributedSampler reshuffles on set_epoch.
        differs = any(
            list(r["epochs"][0]["ids"]) != list(r["epochs"][1]["ids"])
            for r in results
        )
        assert differs, "set_epoch did not change the sampler's ordering"


# ---------------------------------------------------------------------------
# Smoke check: batch shapes are honoured under DDP
# ---------------------------------------------------------------------------


def test_ddp_batch_shapes(result_dir: Path):
    config = {
        "dsm_kwargs": {
            "full_name": "mem/raw_large/train",
            "iterable_dataset_shards": 4,
            "dataloader_kwargs": {
                "batch_size": 4,
                "num_workers": 1,
                "pin_memory": False,
            },
        },
        "run": {"num_epochs": 1, "set_epoch_each_epoch": False},
    }
    results = run_cpu_distributed(
        script_module=ENTRYPOINT,
        world_size=2,
        result_dir=result_dir,
        config=config,
    )
    for r in results:
        for shape in r["epochs"][0]["batch_shapes"]:
            assert shape[1] == EXAMPLE_TOKEN_LEN
            assert 1 <= shape[0] <= 4
