"""Consolidate Lightning FSDP sharded checkpoints to model-only safetensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from huggingface_hub import save_torch_state_dict
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
