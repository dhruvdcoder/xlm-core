"""Unit tests for :mod:`xlm.commands.extract_checkpoint`."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from xlm.commands.extract_checkpoint import extract_checkpoint


class TestExtractCheckpointValidation:
    """Only the up-front config validation is unit-testable.

    The rest of :func:`extract_checkpoint` instantiates Lightning / Hydra
    machinery and is exercised by integration tests.
    """

    def test_missing_post_training_raises(self):
        cfg = OmegaConf.create({"post_training": None})
        with pytest.raises(ValueError, match="post_training"):
            extract_checkpoint(cfg)

    def test_no_destination_raises(self):
        # Both ``model_state_dict_path`` and ``repo_id`` are unset -> nothing
        # to do with the extracted weights.
        cfg = OmegaConf.create(
            {
                "post_training": {
                    "checkpoint_path": "/tmp/fake.ckpt",
                    "apply_ema": False,
                }
            }
        )
        with pytest.raises(ValueError, match="model_state_dict_path"):
            extract_checkpoint(cfg)


def _instantiate_two_step_factory():
    calls = []

    def fake_instantiate(cfg, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {"gc": 1}
        dm = MagicMock()
        dm.tokenizer = MagicMock()
        dm.prepare_data = MagicMock()
        dm.setup = MagicMock()
        return dm

    return fake_instantiate


class TestExtractCheckpointSharded:
    def test_sharded_with_apply_ema_raises(self, tmp_path: Path, monkeypatch):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "shard.distcp").touch()
        (d / "meta.pt").touch()

        monkeypatch.setattr(
            "xlm.commands.extract_checkpoint.hydra.utils.instantiate",
            _instantiate_two_step_factory(),
        )
        monkeypatch.setattr(
            "xlm.commands.extract_checkpoint.print_config_tree",
            lambda *a, **k: None,
        )

        cfg = OmegaConf.create(
            {
                "post_training": {
                    "checkpoint_path": str(d),
                    "apply_ema": True,
                    "model_state_dict_path": str(tmp_path / "out.pth"),
                },
                "seed": None,
                "global_components": {},
                "datamodule": {},
                "lightning_module": {"_target_": "builtins.object"},
            }
        )

        with pytest.raises(ValueError, match="apply_ema=True is not supported"):
            extract_checkpoint(cfg)

    def test_sharded_repo_id_calls_consolidate_and_push(self, tmp_path: Path, monkeypatch):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "shard.distcp").touch()
        (d / "meta.pt").touch()

        consolidated = tmp_path / "m.safetensors"
        consolidated.write_text("x")

        monkeypatch.setattr(
            "xlm.commands.extract_checkpoint.hydra.utils.instantiate",
            _instantiate_two_step_factory(),
        )
        monkeypatch.setattr(
            "xlm.commands.extract_checkpoint.consolidate_model_checkpoint",
            lambda *a, **k: consolidated,
        )
        harness = MagicMock()
        monkeypatch.setattr(
            "xlm.commands.extract_checkpoint.load_model_for_inference",
            lambda *a, **k: (harness, None),
        )
        monkeypatch.setattr(
            "xlm.commands.extract_checkpoint.print_config_tree",
            lambda *a, **k: None,
        )

        cfg = OmegaConf.create(
            {
                "post_training": {
                    "checkpoint_path": str(d),
                    "apply_ema": False,
                    "model_state_dict_path": str(consolidated),
                    "repo_id": "org/model",
                },
                "seed": None,
                "global_components": {},
                "datamodule": {},
                "lightning_module": {"_target_": "builtins.object"},
            }
        )

        extract_checkpoint(cfg)
        harness.push_to_hub.assert_called_once()
        assert harness.push_to_hub.call_args[1]["repo_id"] == "org/model"
