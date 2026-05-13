"""Single-method tests for :class:`xlm.datamodule.DatasetManager`.

Walks every branch of :meth:`DatasetManager.__init__`,
:meth:`DatasetManager.prepare_data`, :meth:`DatasetManager.setup`, and
:meth:`DatasetManager.get_dataloader` in a single Python process.

These tests exercise the real HuggingFace ``datasets`` library, real
``torchdata`` dataloaders, and real on-disk caching -- the ``_download``
method is monkey-patched at the network boundary by the
``patched_download`` fixture so no network I/O is needed.

Datasets, fixtures, and helpers are shared with the integration suite:

* in-memory dataset registry / monkey-patch / processors / collator
  live in :mod:`tests.datamodule_helpers`;
* ``patched_download``, ``manual_cache_dir``, ``simple_collator``, and
  ``dataset_manager_factory`` fixtures live in :mod:`tests.conftest`.

Multi-process / DDP / SLURM / Lightning Trainer paths live under
``tests/integration/datamodule/``.
"""

from __future__ import annotations

from pathlib import Path

import datasets
import pytest
import torch
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from tests.datamodule_helpers import EXAMPLE_TOKEN_LEN
from xlm.datamodule import DatasetManager


# Convenience constants that reflect the in-memory registry contents in
# tests.datamodule_helpers.build_inmem_datasets.
TRAIN_SIZE = 17
VAL_SIZE = 7
TEST_SIZE = 5
LARGE_SIZE = 60


# ---------------------------------------------------------------------------
# Constructor validation -- pure input-checking, no I/O at all
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    """Validates the guard clauses inside :meth:`DatasetManager.__init__`."""

    def test_filter_fn_requires_filter_suffix(self, dataset_manager_factory):
        with pytest.raises(ValueError, match="filter_suffix"):
            dataset_manager_factory(filter_fn=lambda _: True)

    def test_group_processor_requires_iterable_shards(
        self, dataset_manager_factory
    ):
        with pytest.raises(ValueError, match="iterable_dataset_shards"):
            dataset_manager_factory(
                on_the_fly_group_processor=(
                    "tests.datamodule_helpers.pack_with_id"
                )
            )

    def test_per_example_and_group_processors_mutually_exclusive(
        self, dataset_manager_factory
    ):
        with pytest.raises(ValueError, match="cannot both be set"):
            dataset_manager_factory(
                iterable_dataset_shards=2,
                on_the_fly_processor=(
                    "tests.datamodule_helpers.identity_processor"
                ),
                on_the_fly_group_processor=(
                    "tests.datamodule_helpers.pack_with_id"
                ),
            )

    def test_make_infinite_requires_iterable(self, dataset_manager_factory):
        with pytest.raises(ValueError, match="make_infinite"):
            dataset_manager_factory(make_infinite=True)


# ---------------------------------------------------------------------------
# prepare_data branches
# ---------------------------------------------------------------------------


class TestPrepareData:
    """Branches of :meth:`DatasetManager.prepare_data` (lines 985-1044)."""

    def test_no_manual_cache_downloads_and_preprocesses(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        """``use_manual_cache=False`` returns the preprocessed dataset inline."""
        dsm = dataset_manager_factory(use_manual_cache=False)
        ds = dsm.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
        )
        assert ds is not None
        assert len(ds) == TRAIN_SIZE
        assert {"id", "input_ids", "attention_mask", "token_type_ids"}.issubset(
            ds.column_names
        )
        # No on-disk cache should have been written.
        assert not list(manual_cache_dir.iterdir())

    def test_manual_cache_miss_writes_cache_and_returns_when_load_true(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(use_manual_cache=True)
        ds = dsm.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            load=True,
        )
        # Because the cache was missing, prepare_data downloads + caches and
        # the freshly-built dataset is returned regardless of load.
        assert ds is not None
        assert len(ds) == TRAIN_SIZE
        cache_dir = dsm._get_cache_dir(str(manual_cache_dir))
        assert cache_dir.exists()

    def test_manual_cache_hit_load_true_returns_loaded_dataset(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(use_manual_cache=True)
        # First call populates the cache.
        dsm.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
        )
        # Second call hits the cache; load=True returns the loaded dataset.
        ds = dsm.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            load=True,
        )
        assert ds is not None
        assert len(ds) == TRAIN_SIZE

    def test_manual_cache_hit_load_false_returns_none(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(use_manual_cache=True)
        dsm.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
        )
        # On a cache hit, load=False returns None (caller will load later in setup).
        ds = dsm.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            load=False,
        )
        assert ds is None

    def test_rewrite_manual_cache_re_downloads(
        self,
        dataset_manager_factory,
        simple_tokenizer,
        manual_cache_dir,
        patched_download,
    ):
        dsm = dataset_manager_factory(
            use_manual_cache=True, rewrite_manual_cache=True
        )
        # Prime the cache.
        dsm.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
        )
        cache_dir = dsm._get_cache_dir(str(manual_cache_dir))
        assert cache_dir.exists()

        # Mutate the in-memory registry; re-prepare with rewrite_manual_cache=True
        # should pick up the new payload.
        new_ds = datasets.Dataset.from_dict(
            {
                "id": [9000, 9001],
                "text": ["new_a", "new_b"],
                "token_ids": [
                    [10] * EXAMPLE_TOKEN_LEN,
                    [11] * EXAMPLE_TOKEN_LEN,
                ],
            }
        )
        patched_download["mem/raw/train"] = new_ds

        dsm2 = dataset_manager_factory(
            use_manual_cache=True, rewrite_manual_cache=True
        )
        dsm2.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
        )
        loaded = datasets.load_from_disk(cache_dir)
        assert sorted(loaded["id"]) == [9000, 9001]

    def test_filter_fn_reduces_dataset(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(
            use_manual_cache=False,
            # ``drop_first_example`` drops only the row with id=0.
            filter_fn=("tests.datamodule_helpers.drop_first_example"),
            filter_suffix="no_zero",
        )
        ds = dsm.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
        )
        assert ds is not None
        assert 0 not in ds["id"]
        assert len(ds) == TRAIN_SIZE - 1

    def test_filter_suffix_appears_in_cache_dir(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(
            use_manual_cache=True,
            filter_fn=("tests.datamodule_helpers.drop_first_example"),
            filter_suffix="no_zero",
        )
        dsm.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
        )
        cache_dir = dsm._get_cache_dir(str(manual_cache_dir))
        assert "raw_no_zero" in str(cache_dir)
        assert cache_dir.exists()

    @pytest.mark.parametrize("which", ["train", "test"])
    def test_train_test_split(
        self,
        dataset_manager_factory,
        simple_tokenizer,
        manual_cache_dir,
        which: str,
    ):
        dsm = dataset_manager_factory(
            use_manual_cache=False,
            train_test_split={"size": 0.4, "seed": 0, "split": which},
        )
        ds = dsm.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
        )
        assert ds is not None
        # The exact rounding of 17 * 0.4 depends on the HF datasets version;
        # verify that the split is non-empty, not larger than the source, and
        # complements the other side to TRAIN_SIZE.
        dsm_other = dataset_manager_factory(
            use_manual_cache=False,
            train_test_split={
                "size": 0.4,
                "seed": 0,
                "split": "test" if which == "train" else "train",
            },
        )
        other = dsm_other.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
        )
        assert 0 < len(ds) < TRAIN_SIZE
        assert len(ds) + len(other) == TRAIN_SIZE


# ---------------------------------------------------------------------------
# setup branches
# ---------------------------------------------------------------------------


class TestSetup:
    """Branches of :meth:`DatasetManager.setup` (lines 1046-1115)."""

    def test_map_style_basic(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(use_manual_cache=False)
        dsm.setup(
            stage="fit",
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            block_size=EXAMPLE_TOKEN_LEN,
            is_ddp=False,
            rank=0,
            world_size=1,
        )
        assert dsm.dataset is not None
        assert not isinstance(dsm.dataset, torch.utils.data.IterableDataset)
        assert len(dsm.dataset) == TRAIN_SIZE

    def test_iterable_basic(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(
            use_manual_cache=False,
            iterable_dataset_shards=4,
        )
        dsm.setup(
            stage="fit",
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            block_size=EXAMPLE_TOKEN_LEN,
            is_ddp=False,
            rank=0,
            world_size=1,
        )
        assert dsm.is_iterable_dataset
        # Iterable datasets do not expose len(); iterate manually.
        seen = list(dsm.dataset)
        assert len(seen) == TRAIN_SIZE
        assert {row["id"] for row in seen} == set(range(TRAIN_SIZE))

    def test_iterable_with_shuffle_buffer(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(
            use_manual_cache=False,
            iterable_dataset_shards=4,
            shuffle_buffer_size=8,
            shuffle_seed=123,
        )
        dsm.setup(
            stage="fit",
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            block_size=EXAMPLE_TOKEN_LEN,
            is_ddp=False,
            rank=0,
            world_size=1,
        )
        seen_ids = [row["id"] for row in dsm.dataset]
        # Shuffle is non-trivial: order should differ from sorted, but
        # coverage must remain complete.
        assert sorted(seen_ids) == list(range(TRAIN_SIZE))
        assert seen_ids != list(range(TRAIN_SIZE))

    def test_iterable_with_on_the_fly_filter(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(
            use_manual_cache=False,
            iterable_dataset_shards=4,
            on_the_fly_filter_fn=(
                "tests.datamodule_helpers.drop_first_example"
            ),
        )
        dsm.setup(
            stage="fit",
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            block_size=EXAMPLE_TOKEN_LEN,
            is_ddp=False,
            rank=0,
            world_size=1,
        )
        seen_ids = [row["id"] for row in dsm.dataset]
        assert 0 not in seen_ids
        assert len(seen_ids) == TRAIN_SIZE - 1

    def test_iterable_with_on_the_fly_processor(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(
            use_manual_cache=False,
            iterable_dataset_shards=4,
            on_the_fly_processor=(
                "tests.datamodule_helpers.identity_processor"
            ),
        )
        dsm.setup(
            stage="fit",
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            block_size=EXAMPLE_TOKEN_LEN,
            is_ddp=False,
            rank=0,
            world_size=1,
        )
        seen_ids = [row["id"] for row in dsm.dataset]
        assert sorted(seen_ids) == list(range(TRAIN_SIZE))

    def test_iterable_with_group_processor(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        # 17 examples * EXAMPLE_TOKEN_LEN tokens packed into block_size=EXAMPLE_TOKEN_LEN
        # blocks with the same length yields exactly 17 output blocks (no
        # truncation, no padding).
        dsm = dataset_manager_factory(
            full_name="mem/raw/train",
            use_manual_cache=False,
            iterable_dataset_shards=4,
            on_the_fly_group_processor=(
                "tests.datamodule_helpers.pack_with_id"
            ),
            on_the_fly_group_processor_remove_columns=[
                "id",
                "input_ids",
                "attention_mask",
                "token_type_ids",
            ],
        )
        dsm.setup(
            stage="fit",
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            block_size=EXAMPLE_TOKEN_LEN,
            is_ddp=False,
            rank=0,
            world_size=1,
        )
        rows = list(dsm.dataset)
        assert len(rows) == TRAIN_SIZE
        # source_ids on each block records which original ids contributed.
        seen_sources = {sid for row in rows for sid in row["source_ids"]}
        assert seen_sources == set(range(TRAIN_SIZE))

    def test_split_by_node_world_size_1_is_no_op(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        # is_ddp=True but world_size=1 -- the split branch is skipped because
        # of the world_size>1 guard at line 1104.
        dsm = dataset_manager_factory(
            use_manual_cache=False,
            iterable_dataset_shards=4,
        )
        dsm.setup(
            stage="fit",
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            block_size=EXAMPLE_TOKEN_LEN,
            is_ddp=True,
            rank=0,
            world_size=1,
        )
        seen_ids = [row["id"] for row in dsm.dataset]
        assert sorted(seen_ids) == list(range(TRAIN_SIZE))

    def test_setup_skipped_for_unmatched_stage(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        # Default stages=["fit", "validate"]; "test" should leave dataset=None.
        dsm = dataset_manager_factory(use_manual_cache=False)
        dsm.setup(
            stage="test",
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            block_size=EXAMPLE_TOKEN_LEN,
            is_ddp=False,
            rank=0,
            world_size=1,
        )
        assert dsm.dataset is None

    def test_setup_idempotent(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        # Second setup() call should not rebuild self.dataset (guard at line 1072).
        dsm = dataset_manager_factory(use_manual_cache=False)
        dsm.setup(
            stage="fit",
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            block_size=EXAMPLE_TOKEN_LEN,
            is_ddp=False,
            rank=0,
            world_size=1,
        )
        first = dsm.dataset
        dsm.setup(
            stage="fit",
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            block_size=EXAMPLE_TOKEN_LEN,
            is_ddp=False,
            rank=0,
            world_size=1,
        )
        assert dsm.dataset is first


# ---------------------------------------------------------------------------
# get_dataloader cases
# ---------------------------------------------------------------------------


def _setup_for_dataloader(
    dsm: DatasetManager,
    simple_tokenizer,
    manual_cache_dir: Path,
    *,
    is_ddp: bool,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    dsm.setup(
        stage="fit",
        manual_cache_dir=str(manual_cache_dir),
        tokenizer=simple_tokenizer,
        block_size=EXAMPLE_TOKEN_LEN,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size,
    )


class TestGetDataloader:
    """All five branches of :meth:`DatasetManager.get_dataloader` (lines 1117-1223)."""

    def test_train_non_ddp_map_style_case3(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(use_manual_cache=False)
        _setup_for_dataloader(
            dsm, simple_tokenizer, manual_cache_dir, is_ddp=False
        )
        dl = dsm.get_dataloader(
            type="train", is_ddp=False, rank=0, world_size=1
        )
        assert isinstance(dl, StatefulDataLoader)
        batches = list(dl)
        seen = [i for b in batches for i in b["ids"]]
        assert sorted(seen) == list(range(TRAIN_SIZE))

    def test_train_non_ddp_iterable_case4(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(
            use_manual_cache=False, iterable_dataset_shards=4
        )
        _setup_for_dataloader(
            dsm, simple_tokenizer, manual_cache_dir, is_ddp=False
        )
        dl = dsm.get_dataloader(
            type="train", is_ddp=False, rank=0, world_size=1
        )
        assert isinstance(dl, StatefulDataLoader)
        batches = list(dl)
        seen = [i for b in batches for i in b["ids"]]
        assert sorted(seen) == list(range(TRAIN_SIZE))

    def test_train_ddp_iterable_case1_happy_path(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        # world_size=2, num_workers=1, num_shards=4, batch_size=4 -->
        # num_shards/(world*num_workers) = 4/2 = 2 <= batch_size=4 -- OK.
        dsm = dataset_manager_factory(
            full_name="mem/raw_large/train",
            use_manual_cache=False,
            iterable_dataset_shards=4,
            dataloader_kwargs={
                "batch_size": 4,
                "num_workers": 1,
                "pin_memory": False,
            },
        )
        _setup_for_dataloader(
            dsm,
            simple_tokenizer,
            manual_cache_dir,
            is_ddp=True,
            rank=0,
            world_size=2,
        )
        dl = dsm.get_dataloader(
            type="train", is_ddp=True, rank=0, world_size=2
        )
        assert isinstance(dl, StatefulDataLoader)

    def test_train_ddp_iterable_case1_too_many_shards_raises(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        # num_shards=8, world_size=2, num_workers=1 -> per-worker shards = 4,
        # batch_size=2 < 4 -> ValueError (line 1164).
        dsm = dataset_manager_factory(
            full_name="mem/raw_large/train",
            use_manual_cache=False,
            iterable_dataset_shards=8,
            dataloader_kwargs={
                "batch_size": 2,
                "num_workers": 1,
                "pin_memory": False,
            },
        )
        _setup_for_dataloader(
            dsm,
            simple_tokenizer,
            manual_cache_dir,
            is_ddp=True,
            rank=0,
            world_size=2,
        )
        with pytest.raises(ValueError, match="num_shards_per_worker"):
            dsm.get_dataloader(
                type="train", is_ddp=True, rank=0, world_size=2
            )

    def test_train_ddp_iterable_case1_strips_shuffle(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        # Inheriting "shuffle": True should be popped (case 1, line 1150-1154).
        dsm = dataset_manager_factory(
            full_name="mem/raw_large/train",
            use_manual_cache=False,
            iterable_dataset_shards=4,
            dataloader_kwargs={
                "batch_size": 4,
                "num_workers": 1,
                "shuffle": True,
                "pin_memory": False,
            },
        )
        _setup_for_dataloader(
            dsm,
            simple_tokenizer,
            manual_cache_dir,
            is_ddp=True,
            rank=0,
            world_size=2,
        )
        dl = dsm.get_dataloader(
            type="train", is_ddp=True, rank=0, world_size=2
        )
        assert isinstance(dl, StatefulDataLoader)
        assert "shuffle" not in dsm.dataloader_kwargs

    # Case 2 (train + DDP + map-style) cannot be exercised in a single
    # process: ``StatefulDistributedSampler`` calls ``dist.get_world_size()``
    # internally and requires an initialised process group.  See
    # ``tests/integration/datamodule/test_dataset_manager_ddp_cpu.py`` for
    # the equivalent CPU multi-process test.

    @pytest.mark.parametrize("split", ["val", "test", "predict"])
    def test_eval_dataloader_uses_plain_dataloader(
        self,
        dataset_manager_factory,
        simple_tokenizer,
        manual_cache_dir,
        split: str,
    ):
        # The dataloader path depends on ``type`` only, not on the actual
        # split data, so reuse the train dataset for all three eval types.
        dsm = dataset_manager_factory(use_manual_cache=False)
        _setup_for_dataloader(
            dsm, simple_tokenizer, manual_cache_dir, is_ddp=False
        )
        dl = dsm.get_dataloader(
            type=split, is_ddp=False, rank=0, world_size=1
        )
        assert isinstance(dl, DataLoader)
        assert not isinstance(dl, StatefulDataLoader)
        # Eval branch unconditionally rewrites shuffle to False.
        assert dsm.dataloader_kwargs["shuffle"] is False

    def test_invalid_dataloader_type_raises(
        self, dataset_manager_factory, simple_tokenizer, manual_cache_dir
    ):
        dsm = dataset_manager_factory(use_manual_cache=False)
        _setup_for_dataloader(
            dsm, simple_tokenizer, manual_cache_dir, is_ddp=False
        )
        with pytest.raises(ValueError, match="Invalid dataloader type"):
            dsm.get_dataloader(
                type="bogus",  # type: ignore[arg-type]
                is_ddp=False,
                rank=0,
                world_size=1,
            )
