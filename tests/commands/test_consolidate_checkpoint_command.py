"""Tests for ``dream_correction.commands.consolidate_checkpoint`` (project entrypoint)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

_LEARNED_CORRECTORS_ROOT = Path(__file__).resolve().parents[4]
if str(_LEARNED_CORRECTORS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEARNED_CORRECTORS_ROOT))

from dream_correction.commands import consolidate_checkpoint  # noqa: E402


class TestConsolidateCheckpointHub:
    def test_skips_upload_when_hub_repo_id_null(self, tmp_path: Path, monkeypatch):
        out = tmp_path / "m.safetensors"
        monkeypatch.setattr(
            "dream_correction.commands.consolidate_model_checkpoint",
            lambda *a, **k: out,
        )
        wrote: list[int] = []
        monkeypatch.setattr(
            "dream_correction.commands.write_model_only_hub_artifacts",
            lambda *a, **k: wrote.append(1),
        )
        pushed: list[int] = []
        monkeypatch.setattr(
            "dream_correction.commands.push_model_only_folder_to_hub",
            lambda *a, **k: pushed.append(1),
        )

        cfg = OmegaConf.create(
            {
                "consolidate_checkpoint": {
                    "sharded_checkpoint_dir": "/fake/sharded",
                    "output": str(out),
                    "max_shard_size": None,
                    "hub": {"repo_id": None},
                },
                "model": {"d_model": 64},
            }
        )
        consolidate_checkpoint(cfg)
        assert not wrote
        assert not pushed

    def test_uploads_when_hub_repo_id_set_single_file_output(
        self, tmp_path: Path, monkeypatch
    ):
        out = tmp_path / "m.safetensors"
        out.write_text("x")
        monkeypatch.setattr(
            "dream_correction.commands.consolidate_model_checkpoint",
            lambda *a, **k: out.resolve(),
        )
        monkeypatch.setattr(
            "dream_correction.commands.write_model_only_hub_artifacts",
            lambda *a, **k: None,
        )
        pushed: list[tuple] = []
        monkeypatch.setattr(
            "dream_correction.commands.push_model_only_folder_to_hub",
            lambda folder, **kw: pushed.append((folder, kw)),
        )

        cfg = OmegaConf.create(
            {
                "consolidate_checkpoint": {
                    "sharded_checkpoint_dir": "/fake/sharded",
                    "output": str(out),
                    "max_shard_size": None,
                    "hub": {"repo_id": "org/n", "commit_message": "hi"},
                },
                "model": {"d_model": 64},
            }
        )
        consolidate_checkpoint(cfg)
        assert len(pushed) == 1
        folder_arg = pushed[0][0]
        assert Path(folder_arg).is_dir()
        assert (Path(folder_arg) / "model.safetensors").is_file()

    def test_sharded_output_uploads_from_output_dir(self, tmp_path: Path, monkeypatch):
        out_dir = tmp_path / "shards"
        out_dir.mkdir()
        index = out_dir / "model.safetensors.index.json"
        index.write_text("{}")
        monkeypatch.setattr(
            "dream_correction.commands.consolidate_model_checkpoint",
            lambda *a, **k: index.resolve(),
        )
        monkeypatch.setattr(
            "dream_correction.commands.write_model_only_hub_artifacts",
            MagicMock(),
        )
        pushed: list[Path] = []
        monkeypatch.setattr(
            "dream_correction.commands.push_model_only_folder_to_hub",
            lambda folder, **kw: pushed.append(Path(folder)),
        )

        cfg = OmegaConf.create(
            {
                "consolidate_checkpoint": {
                    "sharded_checkpoint_dir": "/fake/sharded",
                    "output": str(out_dir),
                    "max_shard_size": "5GB",
                    "hub": {"repo_id": "org/n"},
                },
                "model": {"d_model": 64},
            }
        )
        consolidate_checkpoint(cfg)
        assert pushed == [out_dir.resolve()]
