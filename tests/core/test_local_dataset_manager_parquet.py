"""Tests for LocalDatasetManager parquet loading."""

from __future__ import annotations

from pathlib import Path

import datasets
import pytest

from xlm.datamodule import LocalDatasetManager


class _DummyCollator:
    def __call__(self, examples):
        return examples


def _write_tiny_parquet(path: Path) -> None:
    ds = datasets.Dataset.from_dict(
        {
            "id": ["a", "b"],
            "source_text": ["hallo", "welt"],
            "target_text": ["hello", "world"],
        }
    )
    ds.to_parquet(str(path))


def test_local_dataset_manager_parquet_explicit_data_files(tmp_path: Path):
    parquet_path = tmp_path / "train.parquet"
    _write_tiny_parquet(parquet_path)

    mgr = LocalDatasetManager(
        collator=_DummyCollator(),
        full_name="local/iwslt14_fixture/train",
        full_name_debug="local/iwslt14_fixture/train",
        dataloader_kwargs={
            "batch_size": 2,
            "num_workers": 0,
            "shuffle": False,
            "pin_memory": False,
        },
        ds_type="parquet",
        load_kwargs={"data_files": str(parquet_path)},
        use_manual_cache=False,
        preprocess_function=None,
        on_the_fly_processor=None,
        on_the_fly_group_processor=None,
        columns_to_remove=None,
        stages=["fit"],
        iterable_dataset_shards=None,
        shuffle_buffer_size=None,
    )
    ds = mgr._download()
    assert len(ds) == 2
    assert set(ds.column_names) >= {"id", "source_text", "target_text"}
    assert ds[0]["source_text"] == "hallo"


def test_local_dataset_manager_parquet_derived_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # full_name must be 3 segments for parse_hf_dataset_full_name
    root = tmp_path / "local" / "fixture"
    root.mkdir(parents=True)
    _write_tiny_parquet(root / "train.parquet")
    monkeypatch.chdir(tmp_path)

    mgr = LocalDatasetManager(
        collator=_DummyCollator(),
        full_name="local/fixture/train",
        full_name_debug="local/fixture/train",
        dataloader_kwargs={
            "batch_size": 2,
            "num_workers": 0,
            "shuffle": False,
            "pin_memory": False,
        },
        ds_type="parquet",
        load_kwargs={},
        use_manual_cache=False,
        preprocess_function=None,
        on_the_fly_processor=None,
        on_the_fly_group_processor=None,
        columns_to_remove=None,
        stages=["fit"],
        iterable_dataset_shards=None,
        shuffle_buffer_size=None,
    )
    ds = mgr._download()
    assert len(ds) == 2


def test_local_dataset_manager_rejects_unknown_ds_type():
    with pytest.raises(ValueError, match="Unsupported dataset type"):
        mgr = LocalDatasetManager(
            collator=_DummyCollator(),
            full_name="local/x/train",
            full_name_debug="local/x/train",
            dataloader_kwargs={
                "batch_size": 1,
                "num_workers": 0,
                "shuffle": False,
                "pin_memory": False,
            },
            ds_type="json",
            use_manual_cache=False,
            preprocess_function=None,
            on_the_fly_processor=None,
            on_the_fly_group_processor=None,
            columns_to_remove=None,
            stages=["fit"],
            iterable_dataset_shards=None,
            shuffle_buffer_size=None,
        )
        mgr._download()
