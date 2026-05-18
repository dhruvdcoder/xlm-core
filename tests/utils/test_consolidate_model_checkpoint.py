"""Tests for model checkpoint consolidation and model-only state dict helpers."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file

from xlm.utils.checkpoint_paths import is_consolidatable_lightning_sharded_dir
from xlm.utils.consolidate_model_checkpoint import (
    consolidate_model_checkpoint,
    export_model_only_safetensors_from_consolidated_checkpoint,
    push_model_only_folder_to_hub,
    write_model_only_hub_artifacts,
)
from xlm.utils.hf_hub import _is_safetensors_sharded_index
from xlm.utils.model_state_dict import (
    extract_model_only_from_lightning_state_dict,
    tensor_state_dict_from_checkpoint_dict,
)


def _minimal_linear_ckpt() -> dict:
    layer = nn.Linear(3, 2)
    return {
        "epoch": 0,
        "state_dict": {
            "model.layer.weight": layer.weight.data.clone(),
            "model.layer.bias": layer.bias.data.clone(),
        },
    }


def _minimal_orig_mod_ckpt() -> dict:
    layer = nn.Linear(3, 2)
    return {
        "state_dict": {
            "model._orig_mod.layer.weight": layer.weight.data.clone(),
            "model._orig_mod.layer.bias": layer.bias.data.clone(),
        }
    }


class TestExtractModelOnlyFromLightningStateDict:
    def test_strips_model_prefix(self):
        sd = {
            "model.w": torch.tensor([1.0]),
            "opt": torch.tensor([2.0]),
        }
        out = extract_model_only_from_lightning_state_dict(sd)
        assert set(out.keys()) == {"w"}
        assert torch.equal(out["w"], torch.tensor([1.0]))

    def test_strips_orig_mod_chain(self):
        ckpt = _minimal_orig_mod_ckpt()
        inner = ckpt["state_dict"]
        out = extract_model_only_from_lightning_state_dict(inner)
        assert set(out.keys()) == {"layer.weight", "layer.bias"}


class TestTensorStateDictFromCheckpointDict:
    def test_round_trip_keys(self):
        ckpt = _minimal_linear_ckpt()
        t = tensor_state_dict_from_checkpoint_dict(ckpt)
        assert set(t.keys()) == {"layer.weight", "layer.bias"}

    def test_missing_state_dict_raises(self):
        with pytest.raises(ValueError, match="state_dict"):
            tensor_state_dict_from_checkpoint_dict({"epoch": 0})


class TestExportModelOnlySafetensors:
    def test_single_file(self, tmp_path: Path):
        ckpt = _minimal_linear_ckpt()
        out = tmp_path / "weights.safetensors"
        ret = export_model_only_safetensors_from_consolidated_checkpoint(
            ckpt, out, max_shard_size=None
        )
        assert ret == out.resolve()
        loaded = load_file(str(out))
        assert set(loaded.keys()) == {"layer.weight", "layer.bias"}

    def test_infers_safetensors_suffix(self, tmp_path: Path):
        ckpt = _minimal_linear_ckpt()
        out = tmp_path / "dir" / "mycheck"
        export_model_only_safetensors_from_consolidated_checkpoint(
            ckpt, out, max_shard_size=None
        )
        assert (tmp_path / "dir" / "mycheck.safetensors").is_file()

    def test_sharded_layout(self, tmp_path: Path):
        # Two tensors; tiny max_shard_size forces multiple shards on HF hub.
        ckpt = {
            "state_dict": {
                "model.w1": torch.randn(40),
                "model.w2": torch.randn(40),
            }
        }
        out_dir = tmp_path / "shards"
        ret = export_model_only_safetensors_from_consolidated_checkpoint(
            ckpt, out_dir, max_shard_size=80
        )
        assert ret.parent == out_dir.resolve()
        if _is_safetensors_sharded_index(Path(ret)):
            assert ret.name.endswith(".safetensors.index.json")
            assert len(list(out_dir.glob("*.safetensors"))) >= 1
        else:
            assert ret.name == "model.safetensors"

    def test_sharded_requires_dir_not_file(self, tmp_path: Path):
        ckpt = _minimal_linear_ckpt()
        f = tmp_path / "x.safetensors"
        f.touch()
        with pytest.raises(ValueError, match="directory"):
            export_model_only_safetensors_from_consolidated_checkpoint(
                ckpt, f, max_shard_size=100
            )


class TestConsolidateModelCheckpointValidation:
    def test_requires_meta_pt_and_distcp(self, tmp_path: Path):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "__0_0.distcp").touch()
        assert not is_consolidatable_lightning_sharded_dir(d)

        with pytest.raises(ValueError, match="Lightning FSDP"):
            consolidate_model_checkpoint(d, tmp_path / "out.safetensors")

    def test_requires_distcp(self, tmp_path: Path) -> None:
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "meta.pt").touch()
        assert not is_consolidatable_lightning_sharded_dir(d)

        with pytest.raises(ValueError, match="Lightning FSDP"):
            consolidate_model_checkpoint(d, tmp_path / "out.safetensors")


class TestWriteModelOnlyHubArtifacts:
    def test_writes_config_json_and_full_config_yaml(self, tmp_path: Path):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"model": {"n_layer": 2, "d_model": 64}})
        d = tmp_path / "hub"
        out_p = write_model_only_hub_artifacts(cfg, d)
        assert out_p.name == "config.json"

        import json

        with open(out_p, encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"n_layer": 2, "d_model": 64}
        assert (d / "full_config.yaml").is_file()


class TestPushModelOnlyFolderToHub:
    def test_upload_folder_receive_allow_patterns_and_branch(
        self, tmp_path: Path, monkeypatch
    ):
        from unittest.mock import MagicMock

        mock_api = MagicMock()
        monkeypatch.setattr(
            "xlm.utils.consolidate_model_checkpoint.HfApi",
            lambda **kwargs: mock_api,
        )

        monkeypatch.setenv("HF_HUB_KEY", "hf_fake")

        from huggingface_hub.errors import RevisionNotFoundError

        mock_api.repo_info.side_effect = RevisionNotFoundError("missing")

        (tmp_path / "model.safetensors").write_bytes(b"x")
        (tmp_path / "config.json").write_text("{}")

        push_model_only_folder_to_hub(
            tmp_path,
            repo_id="org/model",
            commit_message="cmsg",
            branch="step-1",
            token="tok",
        )

        mock_api.create_repo.assert_called_once()
        mock_api.create_branch.assert_called_once()
        mock_api.upload_folder.assert_called_once()
        call_kw = mock_api.upload_folder.call_args[1]
        assert call_kw["repo_id"] == "org/model"
        assert call_kw["revision"] == "step-1"
        assert call_kw["commit_message"] == "cmsg"
        assert call_kw["allow_patterns"] is not None
        assert "*.safetensors" in call_kw["allow_patterns"]
