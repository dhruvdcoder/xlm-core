"""Consolidate Lightning FSDP sharded checkpoints to model-only safetensors."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from huggingface_hub import HfApi, save_torch_state_dict
from huggingface_hub.constants import SAFETENSORS_INDEX_FILE, SAFETENSORS_SINGLE_FILE
from safetensors.torch import save_file

from xlm.utils.checkpoint_paths import is_consolidatable_lightning_sharded_dir
from xlm.utils.model_state_dict import tensor_state_dict_from_checkpoint_dict


def export_model_only_safetensors_from_consolidated_checkpoint(
    checkpoint: dict[str, Any],
    output: Path,
    *,
    max_shard_size: Union[str, int, None] = None,
) -> Path:
    """Write model-only weights from a consolidated Lightning checkpoint dict.

    *checkpoint* must follow standard Lightning format with a top-level ``state_dict``
    (e.g. after Lightning's ``_format_checkpoint`` on a loaded distributed checkpoint).

    Args:
        checkpoint: Loaded consolidated checkpoint mapping.
        output: Destination file (single-file mode) or directory (when *max_shard_size* is set).
        max_shard_size: If set (e.g. ``"5GB"`` or ``128`` bytes in HF convention), write
            ``model.safetensors.index.json`` and shards under *output*.

    Returns:
        Path to ``model.safetensors`` or to ``model.safetensors.index.json``.
    """
    output = output.expanduser()
    tensors = tensor_state_dict_from_checkpoint_dict(checkpoint)
    if not tensors:
        raise ValueError("No tensor weights found to export.")

    if max_shard_size is None:
        out_file = (
            output
            if output.suffix == ".safetensors"
            else output.with_suffix(".safetensors")
        )
        if out_file.exists() and out_file.is_dir():
            raise ValueError(
                f"Output path {out_file} is a directory; pass a .safetensors file path "
                "for single-file mode."
            )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(out_file))
        return out_file.resolve()

    if output.exists() and not output.is_dir():
        raise ValueError(
            f"Output path {output} must be a directory when max_shard_size is set."
        )
    output.mkdir(parents=True, exist_ok=True)
    save_torch_state_dict(
        tensors,
        output,
        max_shard_size=max_shard_size,
        safe_serialization=True,
    )
    index = output / SAFETENSORS_INDEX_FILE
    if index.is_file():
        return index.resolve()
    single = output / SAFETENSORS_SINGLE_FILE
    if single.is_file():
        return single.resolve()
    raise RuntimeError(
        f"Expected {SAFETENSORS_INDEX_FILE} or {SAFETENSORS_SINGLE_FILE} under {output}"
    )


logger = logging.getLogger(__name__)

_DEFAULT_HUB_ALLOW_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "full_config.yaml",
    "README.md",
)


def write_model_only_hub_artifacts(cfg: Any, out_dir: Path) -> Path:
    """Write ``config.json`` and ``full_config.yaml`` for a PyTorchModelHubMixin-style upload.

    Mirrors :meth:`xlm.harness.Harness._save_pretrained` config serialization (not weights).

    Args:
        cfg: Hydra ``DictConfig`` with at least a ``model`` subtree.
        out_dir: Directory to write into (created if missing).

    Returns:
        Path to the written ``config.json``.
    """
    from omegaconf import DictConfig, OmegaConf

    if not isinstance(cfg, DictConfig):
        raise TypeError(f"cfg must be a DictConfig, got {type(cfg)}")
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_config = OmegaConf.to_yaml(cfg, resolve=False)
    full_path = out_dir / "full_config.yaml"
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(full_config)
    logger.info("Saved full config to %s", full_path)

    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    logger.info("Saved model config to %s", config_path)
    return config_path.resolve()


def push_model_only_folder_to_hub(
    folder: str | Path,
    *,
    repo_id: str,
    commit_message: str,
    branch: str | None = None,
    private: bool = False,
    create_pr: bool = False,
    token: str | None = None,
    allow_patterns: list[str] | None = None,
) -> None:
    """Upload a folder of model-only artifacts (safetensors + configs) to the Hub.

    Uses :meth:`huggingface_hub.HfApi.create_repo`, optional branch creation, and
    :meth:`huggingface_hub.HfApi.upload_folder`. Does not instantiate a :class:`~xlm.harness.Harness`.
    """
    from huggingface_hub.errors import RevisionNotFoundError

    if token is None:
        token = os.getenv("HF_HUB_KEY")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    if branch is not None and branch != "main":
        try:
            api.repo_info(repo_id=repo_id, repo_type="model", revision=branch)
        except RevisionNotFoundError:
            logger.info("Branch '%s' not found; creating it.", branch)
            api.create_branch(
                repo_id=repo_id,
                repo_type="model",
                branch=branch,
                exist_ok=True,
            )

    patterns = (
        list(allow_patterns)
        if allow_patterns is not None
        else list(_DEFAULT_HUB_ALLOW_PATTERNS)
    )

    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        repo_type="model",
        revision=branch,
        commit_message=commit_message,
        create_pr=create_pr,
        allow_patterns=patterns,
    )
    logger.info("Uploaded model folder to %s", repo_id)


def consolidate_model_checkpoint(
    sharded_checkpoint_dir: str | Path,
    output: str | Path,
    *,
    max_shard_size: Union[str, int, None] = None,
) -> Path:
    """Consolidate a Lightning FSDP sharded directory to model-only safetensors.

    Requires PyTorch >= 2.3 (Lightning uses ``torch.distributed.checkpoint``).

    Args:
        sharded_checkpoint_dir: Folder with ``*.distcp`` shards and ``meta.pt``.
        output: Target ``.safetensors`` path (single-file) or directory (sharded export).
        max_shard_size: Optional HF shard size (e.g. ``"5GB"``) for multi-file layout.

    Returns:
        Path suitable for ``model_only_checkpoint_path`` (weights file or index JSON).
    """
    src = Path(sharded_checkpoint_dir).expanduser().resolve()
    if not is_consolidatable_lightning_sharded_dir(src):
        raise ValueError(
            "Expected a Lightning FSDP sharded checkpoint directory with at least one "
            f"*.distcp shard and meta.pt, got: {src}"
        )

    try:
        from lightning.fabric.utilities.load import _load_distributed_checkpoint
        from lightning.pytorch.utilities.consolidate_checkpoint import (
            _format_checkpoint,
        )
    except ImportError as e:
        raise ImportError(
            "consolidate_model_checkpoint requires `lightning` to be installed."
        ) from e

    raw = _load_distributed_checkpoint(src)
    formatted = _format_checkpoint(raw)
    return export_model_only_safetensors_from_consolidated_checkpoint(
        formatted,
        Path(output),
        max_shard_size=max_shard_size,
    )
