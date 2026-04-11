"""Utilities for Hugging Face Hub integration."""

import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import torch

_logger = logging.getLogger(__name__)
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import (
    PYTORCH_WEIGHTS_NAME,
    SAFETENSORS_INDEX_FILE,
    SAFETENSORS_SINGLE_FILE,
)
from huggingface_hub.utils import validate_repo_id


def _repo_id_from_url(path: str) -> str | None:
    """Extract repo_id from HF Hub URL. Returns None if invalid."""
    try:
        parsed = urlparse(path)
        if "huggingface.co" not in parsed.netloc and parsed.netloc != "hf.co":
            return None
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        if len(parts) == 1 and parts[0]:
            return parts[0]
    except Exception:
        pass
    return None


def _repo_id_from_plain(path: str) -> str | None:
    """Validate and return repo_id for org/repo format. Returns None if invalid."""
    if path.count("/") > 1 or path.startswith(("/", ".")):
        return None
    try:
        validate_repo_id(path)
        return path
    except Exception:
        return None


def repo_id_from_hf_path(path: str) -> str | None:
    """Extract repo_id from HF Hub path (URL or org/repo). Returns None if invalid."""
    if not path or not isinstance(path, str):
        return None
    path = path.strip()
    if path.startswith(("http://", "https://")):
        return _repo_id_from_url(path)
    return _repo_id_from_plain(path)


def _download_sharded_safetensors(
    repo_id: str,
    revision: str,
    token: str | None,
) -> str:
    """Download index + all shard files; return path to the index JSON."""
    index_path = hf_hub_download(
        repo_id=repo_id,
        filename=SAFETENSORS_INDEX_FILE,
        revision=revision,
        token=token,
    )
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"Invalid or empty weight_map in {SAFETENSORS_INDEX_FILE} for {repo_id}."
        )
    for shard_name in sorted(set(weight_map.values())):
        hf_hub_download(
            repo_id=repo_id,
            filename=shard_name,
            revision=revision,
            token=token,
        )
    return index_path


def download_model_weights(
    repo_id: str,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """Download model weights from Hugging Face Hub.

    Tries ``model.safetensors``, then sharded safetensors (``model.safetensors.index.json``
    plus shard files), then ``pytorch_model.bin``.

    Args:
        repo_id: Hugging Face repository ID (e.g., "org/model").
        revision: Git revision (branch, tag, or commit). Defaults to "main".
        token: HF token for private repos. Uses HF_HUB_KEY env if None.

    Returns:
        Path to the downloaded weights file, or to ``model.safetensors.index.json`` when
        weights are sharded (see :func:`load_model_weights_into_model`).

    Raises:
        ValueError: If no supported weight layout exists in the repo.
    """
    if token is None:
        token = os.getenv("HF_HUB_KEY")
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=SAFETENSORS_SINGLE_FILE,
            revision=revision,
            token=token,
        )
    except Exception:
        try:
            return _download_sharded_safetensors(repo_id, revision, token)
        except Exception:
            try:
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    token=token,
                )
            except Exception as e:
                raise ValueError(
                    f"Could not download model weights from {repo_id}. "
                    f"Expected {SAFETENSORS_SINGLE_FILE}, sharded safetensors "
                    f"({SAFETENSORS_INDEX_FILE} + shards), or {PYTORCH_WEIGHTS_NAME}."
                ) from e


def _is_safetensors_sharded_index(path: Path) -> bool:
    return path.is_file() and (
        path.name == SAFETENSORS_INDEX_FILE
        or path.name.endswith(".safetensors.index.json")
    )


def _shard_paths_from_safetensors_index(index_path: str) -> list[Path]:
    """Resolve and validate shard paths next to a Hugging Face safetensors index file."""
    path = Path(index_path)
    with open(path, encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid safetensors index (no weight_map): {index_path}")
    shard_names = sorted(set(weight_map.values()))
    index_dir = path.parent
    out: list[Path] = []
    for shard_name in shard_names:
        shard_path = index_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"Missing safetensors shard '{shard_name}' next to index {index_path}. "
                "Re-run download_model_weights for this repo."
            )
        out.append(shard_path)
    return out


def _state_dict_from_safetensors_index(
    index_path: str,
    map_location: str = "cpu",
) -> dict:
    """Merge shard files referenced by a Hugging Face safetensors index into one state dict.

    Warning: peak host RAM is ~2× model size. Prefer
    :func:`_load_sharded_safetensors_into_model` for large sharded checkpoints.
    """
    from safetensors.torch import load_file

    state_dict: dict = {}
    for shard_path in _shard_paths_from_safetensors_index(index_path):
        state_dict.update(load_file(str(shard_path), device=map_location))
    return state_dict


def _load_sharded_safetensors_into_model(
    model: torch.nn.Module,
    index_path: str,
    map_location: str = "cpu",
    strict: bool = True,
) -> None:
    """Load sharded safetensors without materializing a full merged state dict in RAM.

    Loads one shard at a time into ``model`` (``strict=False`` per shard), then enforces
    ``strict`` by comparing checkpoint keys to ``model.state_dict()`` keys. Peak RAM is
    roughly model weights plus one shard instead of model plus full checkpoint.
    """
    from safetensors.torch import load_file

    shard_keys_union: set[str] = set()
    for shard_path in _shard_paths_from_safetensors_index(index_path):
        shard_sd = load_file(str(shard_path), device=map_location)
        shard_keys_union.update(shard_sd.keys())
        model.load_state_dict(shard_sd, strict=False)
        del shard_sd
    if strict:
        model_keys = set(model.state_dict().keys())
        missing = model_keys - shard_keys_union
        unexpected = shard_keys_union - model_keys
        if missing or unexpected:
            raise RuntimeError(
                f"Sharded load strict check failed: missing_keys={sorted(missing)!s}, "
                f"unexpected_keys={sorted(unexpected)!s}"
            )


def load_model_state_dict_from_file(
    checkpoint_path: str,
    map_location: str = "cpu",
    weights_only: bool = True,
):
    """Load model state dict from a checkpoint file (safetensors or pickle).

    For .safetensors uses load_file directly (no metadata validation).
    For ``model.safetensors.index.json`` merges all referenced shards into one dict
    (high peak RAM for large models; prefer loading via ``load_model_weights_into_model``).
    For .bin/.pt uses torch.load.

    Args:
        checkpoint_path: Path to model weights, index JSON, or pytorch_model.bin.
        map_location: Device to load tensors to.
        weights_only: If True, use weights_only for pickle (PyTorch >= 1.13).

    Returns:
        State dict for model.load_state_dict().
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint file found at '{checkpoint_path}'")

    if _is_safetensors_sharded_index(path):
        return _state_dict_from_safetensors_index(checkpoint_path, map_location=map_location)

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(checkpoint_path, device=map_location)
    # Pickle format (.bin, .pt)
    load_kwargs: dict = {"map_location": map_location}
    if weights_only:
        load_kwargs["weights_only"] = True
    try:
        return torch.load(checkpoint_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        return torch.load(checkpoint_path, **load_kwargs)


def load_model_weights_into_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    map_location: str = "cpu",
    strict: bool = True,
    weights_only: bool = True,
):
    """Load weights from checkpoint into model. Aligns with harness and hub_mixin.

    For .safetensors uses safetensors.torch.load_model (handles tensor sharing).
    For sharded safetensors (``model.safetensors.index.json``) loads one shard at a
    time to limit peak CPU memory (avoids holding a full merged state dict).
    For .bin/.pt uses model.load_state_dict(torch.load(...)).

    Args:
        model: The model to load weights into.
        checkpoint_path: Path to weights file, safetensors index, or pytorch_model.bin.
        map_location: Device to load tensors to.
        strict: Whether to enforce exact key match (safetensors) or pass to load_state_dict.
        weights_only: If True, use weights_only for pickle (PyTorch >= 1.13).
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint file found at '{checkpoint_path}'")

    if _is_safetensors_sharded_index(path):
        _load_sharded_safetensors_into_model(
            model, checkpoint_path, map_location=map_location, strict=strict
        )
        return

    if path.suffix == ".safetensors":
        from safetensors.torch import load_model as load_model_as_safetensor

        load_model_as_safetensor(
            model, checkpoint_path, strict=strict, device=map_location
        )
        return
    # Pickle format (.bin, .pt)
    state_dict = load_model_state_dict_from_file(
        checkpoint_path, map_location=map_location, weights_only=weights_only
    )
    result = model.load_state_dict(state_dict, strict=strict)
    if not strict and (result.missing_keys or result.unexpected_keys):
        _logger.warning(
            f"load_state_dict: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
