"""Utilities for Hugging Face Hub integration."""

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import torch

_logger = logging.getLogger(__name__)
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import PYTORCH_WEIGHTS_NAME, SAFETENSORS_SINGLE_FILE
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


def download_model_weights(
    repo_id: str,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """Download model weights from Hugging Face Hub.

    Tries model.safetensors first, then pytorch_model.bin.

    Args:
        repo_id: Hugging Face repository ID (e.g., "org/model").
        revision: Git revision (branch, tag, or commit). Defaults to "main".
        token: HF token for private repos. Uses HF_HUB_KEY env if None.

    Returns:
        Path to the downloaded weights file.

    Raises:
        ValueError: If neither model.safetensors nor pytorch_model.bin exists.
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
            return hf_hub_download(
                repo_id=repo_id,
                filename=PYTORCH_WEIGHTS_NAME,
                revision=revision,
                token=token,
            )
        except Exception as e:
            raise ValueError(
                f"Could not download model weights from {repo_id}. "
                f"Expected {SAFETENSORS_SINGLE_FILE} or {PYTORCH_WEIGHTS_NAME}."
            ) from e


def load_model_state_dict_from_file(
    checkpoint_path: str,
    map_location: str = "cpu",
    weights_only: bool = True,
):
    """Load model state dict from a checkpoint file (safetensors or pickle).

    For .safetensors uses load_file directly (no metadata validation).
    For .bin/.pt uses torch.load.

    Args:
        checkpoint_path: Path to model.safetensors or pytorch_model.bin.
        map_location: Device to load tensors to.
        weights_only: If True, use weights_only for pickle (PyTorch >= 1.13).

    Returns:
        State dict for model.load_state_dict().
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint file found at '{checkpoint_path}'")

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
    For .bin/.pt uses model.load_state_dict(torch.load(...)).

    Args:
        model: The model to load weights into.
        checkpoint_path: Path to model.safetensors or pytorch_model.bin.
        map_location: Device to load tensors to.
        strict: Whether to enforce exact key match (safetensors) or pass to load_state_dict.
        weights_only: If True, use weights_only for pickle (PyTorch >= 1.13).
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint file found at '{checkpoint_path}'")

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
