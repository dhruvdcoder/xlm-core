"""Unit tests for :mod:`xlm.utils.saving_utils`."""

import json
import logging
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from xlm.utils.saving_utils import (
    process_state_dict,
    save_predictions,
    save_predictions_from_dataloader,
    tags_to_str,
)


class TestProcessStateDict:
    def test_default_keeps_all_keys(self):
        sd = OrderedDict([("a", 1), ("b", 2)])
        out = process_state_dict(sd)
        assert dict(out) == {"a": 1, "b": 2}

    def test_symbols_strips_prefix(self):
        sd = OrderedDict(
            [("model.linear.weight", 1), ("model.linear.bias", 2)]
        )
        out = process_state_dict(sd, symbols=len("model."))
        assert list(out.keys()) == ["linear.weight", "linear.bias"]

    def test_single_string_exception_drops_matching(self):
        sd = OrderedDict([("model.a", 1), ("ema.b", 2), ("model.c", 3)])
        out = process_state_dict(sd, exceptions="ema")
        assert "ema.b" not in out
        assert set(out.keys()) == {"model.a", "model.c"}

    def test_list_of_string_exceptions(self):
        sd = OrderedDict(
            [("model.a", 1), ("ema.b", 2), ("loss.c", 3), ("model.d", 4)]
        )
        out = process_state_dict(sd, exceptions=["ema", "loss"])
        assert set(out.keys()) == {"model.a", "model.d"}


class TestSavePredictionsFromDataloader:
    def test_writes_json(self, tmp_path: Path):
        path = tmp_path / "out.json"
        # Each "batch" must contain values that support .tolist().
        predictions = [
            {
                "logits": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "ids": torch.tensor([[10, 20], [30, 40]]),
            }
        ]
        save_predictions_from_dataloader(predictions, path)
        data = json.loads(path.read_text())
        # Two examples per batch -> two top-level entries.
        assert len(data) == 2
        # Each entry has both fields.
        for entry in data.values():
            assert "logits" in entry and "ids" in entry

    def test_writes_json_with_names(self, tmp_path: Path):
        path = tmp_path / "out.json"
        predictions = [
            {
                "names": ["alpha", "beta"],
                "ids": torch.tensor([[1, 2], [3, 4]]),
            }
        ]
        save_predictions_from_dataloader(predictions, path)
        data = json.loads(path.read_text())
        assert "alpha" in data and "beta" in data

    def test_writes_csv(self, tmp_path: Path):
        path = tmp_path / "out.csv"
        predictions = [{"ids": torch.tensor([[1, 2], [3, 4]])}]
        save_predictions_from_dataloader(predictions, path)
        assert path.exists() and path.stat().st_size > 0

    def test_unsupported_extension_raises(self, tmp_path: Path):
        path = tmp_path / "out.pkl"
        with pytest.raises(NotImplementedError):
            save_predictions_from_dataloader([{"x": torch.zeros(1, 1)}], path)


class TestSavePredictions:
    def test_empty_predictions_logs_and_returns(self, tmp_path: Path, caplog):
        caplog.set_level(logging.WARNING, logger="xlm.utils.saving_utils")
        save_predictions([], str(tmp_path))
        assert any("empty" in r.message.lower() for r in caplog.records)
        # No predictions/ directory should have been created.
        assert not (tmp_path / "predictions").exists()

    def test_invalid_output_format_raises(self, tmp_path: Path):
        with pytest.raises(NotImplementedError):
            save_predictions(
                [{"x": torch.zeros(1, 1)}], str(tmp_path), output_format="xml"
            )

    def test_dict_predictions_writes_single_file(self, tmp_path: Path):
        predictions = [
            {"ids": torch.tensor([[1, 2], [3, 4]])},
        ]
        save_predictions(predictions, str(tmp_path), output_format="json")
        out = tmp_path / "predictions" / "predictions.json"
        assert out.exists()

    def test_list_of_lists_writes_per_dataloader_files(self, tmp_path: Path):
        predictions = [
            [{"ids": torch.tensor([[1, 2], [3, 4]])}],
            [{"ids": torch.tensor([[5, 6], [7, 8]])}],
        ]
        save_predictions(predictions, str(tmp_path), output_format="json")
        assert (tmp_path / "predictions" / "predictions_0.json").exists()
        assert (tmp_path / "predictions" / "predictions_1.json").exists()

    def test_list_with_empty_inner_logs_warning(
        self, tmp_path: Path, caplog
    ):
        caplog.set_level(logging.WARNING, logger="xlm.utils.saving_utils")
        predictions = [[], [{"ids": torch.tensor([[5, 6]])}]]
        save_predictions(predictions, str(tmp_path), output_format="json")
        assert any("empty" in r.message.lower() for r in caplog.records)
        # The non-empty inner is still written.
        assert (tmp_path / "predictions" / "predictions_1.json").exists()

    def test_unsupported_predictions_format_raises(self, tmp_path: Path):
        with pytest.raises(Exception):
            save_predictions(
                ["not a dict or list"], str(tmp_path), output_format="json"
            )


class TestTagsToStr:
    def test_returns_empty_when_tags_missing(self):
        cfg = OmegaConf.create({"unrelated": True})
        assert tags_to_str(cfg) == ""

    def test_formats_tags(self):
        cfg = OmegaConf.create({"tags": {"model": "arlm", "lr": 0.001}})
        out = tags_to_str(cfg)
        # Either order is fine but both pieces must be present.
        assert "model=arlm" in out
        assert "lr=0.001" in out
        assert "__" in out

    def test_custom_location(self):
        cfg = OmegaConf.create({"meta": {"k": 1}})
        assert tags_to_str(cfg, location="meta") == "k=1"
