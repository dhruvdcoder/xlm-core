"""Tests for Hugging Face ``full_name`` parsing in :mod:`xlm.datamodule`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from xlm.datamodule import DatasetManager, parse_hf_dataset_full_name


def test_parse_three_part_full_name() -> None:
    assert parse_hf_dataset_full_name("openai/gsm8k/test") == (
        "openai/gsm8k",
        None,
        "test",
    )


def test_parse_four_part_full_name_with_config() -> None:
    assert parse_hf_dataset_full_name("openai/gsm8k/main/test") == (
        "openai/gsm8k",
        "main",
        "test",
    )


def test_parse_invalid_full_name_raises() -> None:
    with pytest.raises(ValueError, match="repo/dataset/split"):
        parse_hf_dataset_full_name("only/two")


def _minimal_dataset_manager(
    simple_collator, full_name: str
) -> DatasetManager:
    return DatasetManager(
        collator=simple_collator,
        full_name=full_name,
        full_name_debug=full_name,
        dataloader_kwargs={
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
        },
    )


def test_dataset_manager_download_passes_config_name(simple_collator) -> None:
    dsm = _minimal_dataset_manager(
        simple_collator, full_name="openai/gsm8k/main/test"
    )
    fake_ds = MagicMock()
    with patch("xlm.datamodule.datasets.load_dataset", return_value=fake_ds) as load:
        dsm._download()
    load.assert_called_once_with(
        "openai/gsm8k",
        "main",
        split="test",
        num_proc=None,
        token=None,
    )


def test_dataset_manager_download_three_part_unchanged(simple_collator) -> None:
    dsm = _minimal_dataset_manager(simple_collator, full_name="mem/raw/test")
    fake_ds = MagicMock()
    with patch("xlm.datamodule.datasets.load_dataset", return_value=fake_ds) as load:
        dsm._download()
    load.assert_called_once_with(
        "mem/raw",
        split="test",
        num_proc=None,
        token=None,
    )
