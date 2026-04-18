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
# mem/raw_varlen/train shard layout (see tests.datamodule_helpers.VARLEN_SHARD_LENGTHS).
# Per-shard packed-block counts under pack_sequences(use_bos=True,
# drop_last=True, block_size=8): 3 / 4 / 3 / 6.
VARLEN_BLOCKS_PER_SHARD = (3, 4, 3, 6)


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
# Iterable + DDP + pack_sequences group processor: hypothesis check
# ---------------------------------------------------------------------------


class TestIterableDdpPackSequencesUnevenBatches:
    """Hypothesis: ``pack_sequences`` as ``on_the_fly_group_processor`` plus
    ``shuffle()`` and ``split_dataset_by_node()`` can produce **unequal
    per-rank batch counts**, which would deadlock real DDP training the
    moment a collective (allreduce, barrier, ...) is executed in the
    train loop.

    The CPU runner does not run any collective between batches, so it
    cannot itself hang -- but each rank simply iterating its dataloader
    to completion lets us observe per-rank batch counts directly and
    confirm or refute the hypothesis without any chance of a stuck
    pytest job.

    Why the imbalance is unavoidable here
    -------------------------------------

    The pipeline (see :meth:`xlm.datamodule.DatasetManager.setup`) is
    ``to_iterable_dataset(num_shards=4)`` -> ``shuffle(buffer_size,
    seed)`` -> ``map(pack_sequences, batched=True)`` ->
    ``split_dataset_by_node(world_size=2)``.  Each step's effect on
    *per-shard packed-block counts*:

    * ``to_iterable_dataset`` chunks the rows of
      ``mem/raw_varlen/train`` (see
      :data:`tests.datamodule_helpers.VARLEN_SHARD_LENGTHS`)
      contiguously, so each of the 4 shards receives one of the
      ``[4]*4 / [6]*4 / [4]*4 / [10]*4`` groups.
    * ``shuffle`` permutes the shard order (HF
      ``IterableDataset.shuffle`` calls ``shuffle_data_sources`` on the
      underlying ``ExamplesIterable``) but does not change which input
      examples live in which shard.
    * ``pack_sequences(block_size=8, use_bos=True, drop_last=True)`` is
      applied via ``ds.map(batched=True)`` per shard, producing the
      multiset ``{3, 4, 3, 6}`` packed blocks per shard
      (``VARLEN_BLOCKS_PER_SHARD``).
    * ``split_dataset_by_node`` ultimately calls
      ``shard_data_sources(num_shards=world_size, index=rank,
      contiguous=False)``: ranks pick shards round-robin (strided) from
      the post-shuffle order.

    With 4 shards, ``world_size=2`` and ``contiguous=False``, every
    permutation of ``{3, 4, 3, 6}`` lands on rank 0 / rank 1 as one of
    the unordered pairings ``{3+3, 4+6}``, ``{3+4, 3+6}`` or
    ``{3+6, 4+3}`` -- summing to ``{6, 10}``, ``{7, 9}`` or ``{9, 7}``
    respectively.  **None of them sum to ``{8, 8}``.**  No choice of
    ``shuffle_seed`` can balance this multiset across two ranks, so the
    per-rank batch counts must differ.  ``num_workers=2`` and
    ``batch_size=1`` simply make the per-rank batch count equal to the
    per-rank packed-block count.

    The assertion that all ranks observe the same number of batches
    therefore *fails*, demonstrating the hypothesis.  The test is
    marked ``xfail(strict=True)`` so the suite stays green while the
    issue is open and will flip to a hard failure the day the data
    pipeline learns to balance its output across ranks.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "pack_sequences group processor + variable-length sources + "
            "split_dataset_by_node yields uneven per-rank batch counts "
            "(no shuffle_seed can balance the {3,4,3,6} packed-block "
            "multiset across 2 ranks); this would hang real DDP training "
            "on the next collective."
        ),
    )
    def test_pack_sequences_uneven_per_rank_batch_counts(
        self, result_dir: Path
    ):
        block_size = 8
        config = {
            "dsm_kwargs": {
                "full_name": "mem/raw_varlen/train",
                "iterable_dataset_shards": 4,
                # Skip the per-example preprocess so token_ids survives
                # untouched into the group processor.
                "preprocess_function": None,
                "columns_to_keep": None,
                "on_the_fly_group_processor": "xlm.datamodule.pack_sequences",
                "on_the_fly_group_processor_kwargs": {
                    "drop_last": True,
                    "use_bos": True,
                },
                # pack_sequences changes the batch size, so HF .map() needs
                # the input columns dropped or it will try to merge them
                # with the (shorter) outputs.
                "on_the_fly_group_processor_remove_columns": [
                    "id",
                    "text",
                    "token_ids",
                ],
                "shuffle_buffer_size": 4,
                "shuffle_seed": 17,
                "dataloader_kwargs": {
                    "batch_size": 1,
                    "num_workers": 2,
                    "pin_memory": False,
                },
            },
            "setup_kwargs": {"block_size": block_size},
            "run": {"num_epochs": 1, "set_epoch_each_epoch": False},
        }
        results = run_cpu_distributed(
            script_module=ENTRYPOINT,
            world_size=2,
            result_dir=result_dir,
            config=config,
            timeout=180.0,
        )
        assert len(results) == 2
        assert all(r["is_iterable_dataset"] for r in results)

        # Every emitted batch must hold a single packed block of the
        # configured block_size -- if pack_sequences silently emitted
        # short / padded blocks the next assertion would be uninterpretable.
        for r in results:
            for shape in r["epochs"][0]["batch_shapes"]:
                assert shape == [1, block_size], (
                    f"rank {r['rank']}: unexpected batch shape {shape} "
                    f"(expected [1, {block_size}])"
                )

        per_rank_batches = [
            len(r["epochs"][0]["batch_shapes"]) for r in results
        ]

        # Hypothesis: the {3,4,3,6} packed-block multiset cannot be
        # balanced across 2 ranks under round-robin (strided) shard
        # assignment, regardless of shuffle_seed (see class docstring).
        # We assert what the *correct* behaviour should be -- equal
        # per-rank batch counts -- so this fails today and the xfail
        # above records it.
        assert per_rank_batches[0] == per_rank_batches[1], (
            f"per-rank batch counts differ: rank0={per_rank_batches[0]}, "
            f"rank1={per_rank_batches[1]}; per-shard packed-block layout "
            f"{VARLEN_BLOCKS_PER_SHARD} sums to 16 but no permutation "
            "balances {3,4,3,6} into equal pairs under round-robin "
            "split_dataset_by_node."
        )


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
