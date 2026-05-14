"""Unified model loading for inference across commands.

This module provides a single, consistent interface for loading models
for inference tasks (generation, evaluation, push to hub, demos).
"""

import contextlib
import os
from typing import Any, Optional, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers.modeling_utils import no_init_weights

from xlm.harness import Harness
from xlm.utils.hf_hub import (
    download_model_weights,
    load_model_weights_into_model,
    repo_id_from_hf_path,
)
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def _normalize_optional_str(val: Any) -> Optional[str]:
    """Return a non-empty string or None.

    Hydra CLI ``...=null`` / YAML ``null`` sometimes surface as the literal
    string ``\"None\"``; treat those like missing so Hub / fallbacks work.
    """
    if val is None:
        return None
    try:
        if OmegaConf.is_missing(val):
            return None
    except (ValueError, TypeError, AttributeError):
        pass
    if not isinstance(val, str):
        val = str(val)
    s = val.strip()
    if not s or s.lower() in ("none", "null", "~"):
        return None
    return s


_STR_TO_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_init_dtype(cfg: DictConfig) -> Optional[torch.dtype]:
    """Read ``init_dtype`` from config (if present) and resolve to a torch dtype."""
    raw = OmegaConf.select(cfg, "init_dtype", default=None)
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw not in _STR_TO_DTYPE:
            raise ValueError(
                f"Unsupported init_dtype '{raw}'. Choose from {list(_STR_TO_DTYPE)}."
            )
        return _STR_TO_DTYPE[raw]
    return raw


@contextlib.contextmanager
def _default_dtype_ctx(dtype: Optional[torch.dtype]):
    """Temporarily set ``torch.set_default_dtype`` if *dtype* is not None."""
    if dtype is None:
        yield
        return
    orig = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(orig)


def load_model_for_inference(
    cfg: DictConfig,
    datamodule: Any,
    tokenizer: Any,
    *,
    config_prefix: str,
    manual_ema_restore: bool = False,
    move_to_device: Optional[str] = None,
    set_eval_mode: bool = False,
    enable_hub_support: bool = True,
    checkpoint_fallback_dir: Optional[str] = None,
    allow_random_init: bool = False,
) -> tuple[Harness, Optional[str]]:
    """Load and prepare a model for inference tasks.

    This function provides a unified interface for loading models across different
    commands (generate, eval, push_to_hub, cli_demo). It supports loading from:
    - Full Lightning checkpoints (includes optimizer state, etc.)
    - Model-only checkpoints (just model weights)
    - Hugging Face Hub repositories
    - Fallback to best.ckpt or last.ckpt

    Args:
        cfg: Hydra config containing model and checkpoint configuration.
        datamodule: Datamodule instance for model instantiation.
        tokenizer: Tokenizer instance for model instantiation.
        config_prefix: Prefix for config keys. Examples:
            - "generation" looks for cfg.generation.ckpt_path
            - "eval" looks for cfg.eval.checkpoint_path
            - "" (empty) looks for cfg.hub_checkpoint_path (top-level)
        manual_ema_restore: If True, pass manual_ema_restore=True to model loading.
            Used when you need to manually control EMA weight restoration.
        move_to_device: Device to move model to ("cuda", "cpu", or None).
            If None, model stays on default device (trainer handles this for eval).
        set_eval_mode: If True, call model.eval() after loading.
            Set to False when using Lightning Trainer (trainer handles this).
        enable_hub_support: If True, support loading from hub.repo_id.
            Set to False for commands with different hub key structures.
        checkpoint_fallback_dir: Directory to search for best.ckpt/last.ckpt
            if no explicit checkpoint is provided. Used by eval command.
        allow_random_init: If True, allow instantiating a randomly initialized model
            when no checkpoint is found. Default False for safety.

    Returns:
        Tuple of (lightning_module, checkpoint_path) where:
        - lightning_module: The loaded Harness model ready for inference
        - checkpoint_path: Path to the full checkpoint (or None if using model-only
          checkpoint or random init). Used by eval to pass to trainer.

    Raises:
        ValueError: If no checkpoint is found and allow_random_init=False.
        ValueError: If checkpoint file doesn't exist.

    Examples:
        # Generation: Load from checkpoint with HF Hub support
        >>> module, _ = load_model_for_inference(
        ...     cfg, datamodule, tokenizer,
        ...     config_prefix="generation",
        ...     manual_ema_restore=True,
        ...     move_to_device="cuda",
        ...     set_eval_mode=True,
        ...     enable_hub_support=True,
        ... )

        # Evaluation: Return checkpoint path for trainer
        >>> module, ckpt_path = load_model_for_inference(
        ...     cfg, datamodule, tokenizer,
        ...     config_prefix="eval",
        ...     checkpoint_fallback_dir=cfg.checkpointing_dir,
        ... )
        >>> trainer.validate(module, datamodule, ckpt_path=ckpt_path)

        # Push to Hub: Top-level config keys
        >>> module, _ = load_model_for_inference(
        ...     cfg, datamodule, tokenizer,
        ...     config_prefix="",
        ...     manual_ema_restore=True,
        ...     move_to_device="cuda",
        ...     set_eval_mode=True,
        ...     enable_hub_support=False,
        ... )
    """
    torch.set_float32_matmul_precision("medium")

    # Step 1: Determine full checkpoint path
    full_ckpt_path = _get_full_checkpoint_path(
        cfg, config_prefix, checkpoint_fallback_dir
    )

    # Step 2: Determine model-only checkpoint path
    model_only_ckpt_path = _get_model_only_checkpoint_path(
        cfg, config_prefix, enable_hub_support, full_ckpt_path
    )

    # Step 3: Validate that we have at least one checkpoint (or allow_random_init)
    if full_ckpt_path is None and model_only_ckpt_path is None:
        if not allow_random_init:
            raise ValueError(
                "No checkpoint provided for model loading. "
                "Please provide one of: "
                f"- {config_prefix + '.' if config_prefix else ''}ckpt_path / checkpoint_path "
                f"- {config_prefix + '.' if config_prefix else ''}model_only_checkpoint_path "
                + ("- hub.repo_id " if enable_hub_support else "")
                + "Or set allow_random_init=True to use randomly initialized weights."
            )
        logger.warning(
            "No checkpoint provided and allow_random_init=True. "
            "Instantiating model with random initialization."
        )

    # Step 4: Load or instantiate the model
    module_cls = hydra.utils.get_class(cfg.lightning_module._target_)
    return_ckpt_path: Optional[str] = None

    init_dtype = _resolve_init_dtype(cfg)

    if full_ckpt_path is not None:
        # Load from full checkpoint
        logger.info(f"Loading model from full checkpoint: {full_ckpt_path}")
        load_kwargs = {
            "checkpoint_path": full_ckpt_path,
            "tokenizer": tokenizer,
            "datamodule": datamodule,
            "cfg": cfg,
        }
        if manual_ema_restore:
            load_kwargs["manual_ema_restore"] = True

        with _default_dtype_ctx(init_dtype):
            lightning_module = module_cls.load_from_checkpoint(**load_kwargs)
        return_ckpt_path = full_ckpt_path
    else:
        # Instantiate new model
        logger.info("Instantiating new lightning module")
        instantiate_kwargs = {
            "tokenizer": tokenizer,
            "datamodule": datamodule,
            "cfg": cfg,
            "_recursive_": False,
        }
        if manual_ema_restore:
            instantiate_kwargs["manual_ema_restore"] = True

        skip_init = (
            OmegaConf.select(cfg, "skip_init_weights", default=False)
            and model_only_ckpt_path is not None
        )
        if skip_init:
            logger.info(
                "Skipping weight initialization (skip_init_weights=True, "
                "pretrained weights will be loaded)"
            )
        skip_ctx = no_init_weights() if skip_init else contextlib.nullcontext()
        # no_init_weights() will temporarily disable all random init operations which will save significant time in loading large models when we know that we will be loading pretrained weights anyway so no need to random init.
        # FUTURE: We can only use no_init_weights() but not init_empty_weights()
        # because lightning trainer expects real weights to be present.
        # A major feature would be to support empty init weights but that would require making changes to the trainer to support it.

        with _default_dtype_ctx(init_dtype), skip_ctx:
            lightning_module = hydra.utils.instantiate(
                cfg.lightning_module, **instantiate_kwargs
            )

    lightning_module = cast(Harness, lightning_module)

    # Step 5: Load model-only checkpoint if provided
    if model_only_ckpt_path is not None:
        logger.info(f"Loading model weights from: {model_only_ckpt_path}")
        load_model_weights_into_model(
            lightning_module.model,
            model_only_ckpt_path,
            map_location="cpu",
            strict=True,
            weights_only=True,
        )
        logger.warning(
            f"Loaded weights for `model` from {model_only_ckpt_path}. "
            "Make sure that the model weights were saved with EMA applied if needed."
        )
        # When using model-only checkpoint, don't return checkpoint path
        # (prevents Lightning from trying to reload)
        return_ckpt_path = None

    # Step 6: Post-processing - move to device and set eval mode
    if move_to_device is not None:
        logger.info(f"Moving model to device: {move_to_device}")
        lightning_module = lightning_module.to(move_to_device)

    if set_eval_mode:
        logger.info("Setting model to eval mode")
        lightning_module.eval()

    return lightning_module, return_ckpt_path


def _get_full_checkpoint_path(
    cfg: DictConfig,
    config_prefix: str,
    checkpoint_fallback_dir: Optional[str],
) -> Optional[str]:
    """Determine the full checkpoint path from config.

    Args:
        cfg: Hydra config.
        config_prefix: Config prefix ("generation", "eval", or "").
        checkpoint_fallback_dir: Directory to search for fallback checkpoints.

    Returns:
        Full checkpoint path or None if not found.
    """
    # Build the config path to check
    if config_prefix:
        # Try both ckpt_path and checkpoint_path variants
        ckpt_path = _normalize_optional_str(
            OmegaConf.select(cfg, f"{config_prefix}.ckpt_path", default=None)
        )
        if ckpt_path is None:
            ckpt_path = _normalize_optional_str(
                OmegaConf.select(
                    cfg, f"{config_prefix}.checkpoint_path", default=None
                )
            )
    else:
        # Top-level: check hub_checkpoint_path
        ckpt_path = _normalize_optional_str(cfg.get("hub_checkpoint_path", None))

    if ckpt_path is not None:
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint path does not exist: {ckpt_path}")
        return ckpt_path

    # Try fallback directory if provided
    if checkpoint_fallback_dir is not None:
        # Try best.ckpt first
        best_path = os.path.join(checkpoint_fallback_dir, "best.ckpt")
        if os.path.isfile(best_path):
            logger.info(f"Using fallback checkpoint: {best_path}")
            return best_path

        # Try last.ckpt
        last_path = os.path.join(checkpoint_fallback_dir, "last.ckpt")
        if os.path.isfile(last_path):
            logger.info(f"Using fallback checkpoint: {last_path}")
            return last_path

        logger.info(
            f"No fallback checkpoint found in {checkpoint_fallback_dir} "
            "(tried best.ckpt and last.ckpt)"
        )

    return None


def _get_model_only_checkpoint_path(
    cfg: DictConfig,
    config_prefix: str,
    enable_hub_support: bool,
    full_ckpt_path: Optional[str],
) -> Optional[str]:
    """Determine the model-only checkpoint path from config or HF Hub.

    Args:
        cfg: Hydra config.
        config_prefix: Config prefix ("generation", "eval", or "").
        enable_hub_support: Whether to check for hub.repo_id.
        full_ckpt_path: Full checkpoint path (if any). Used for conflict detection.

    Returns:
        Model-only checkpoint path or None if not found.
    """
    # Check for model_only_checkpoint_path in config
    if config_prefix:
        model_only_path = _normalize_optional_str(
            OmegaConf.select(
                cfg, f"{config_prefix}.model_only_checkpoint_path", default=None
            )
        )
    else:
        # Top-level
        model_only_path = _normalize_optional_str(
            cfg.get("model_only_checkpoint_path", None)
        )

    # Check for hub.repo_id if enabled
    hub_repo_id = None
    if enable_hub_support:
        hub_repo_id = _normalize_optional_str(
            OmegaConf.select(cfg, "hub.repo_id", default=None)
        )

    # Conflict detection
    if full_ckpt_path is not None:
        if model_only_path is not None or hub_repo_id is not None:
            logger.error(
                "Full checkpoint and model-only checkpoint cannot both be provided. "
                "Using full checkpoint for model weights."
            )
        return None

    # Handle hub.repo_id
    if hub_repo_id is not None:
        if model_only_path is not None:
            logger.error(
                "hub.repo_id and model_only_checkpoint_path cannot both be provided. "
                "Using hub.repo_id."
            )

        # Download from HF Hub
        repo_id = repo_id_from_hf_path(hub_repo_id) or hub_repo_id
        revision = OmegaConf.select(cfg, "hub.revision", default="main")
        logger.info(
            f"Downloading model weights from Hugging Face Hub: {repo_id}"
        )

        return download_model_weights(
            repo_id=repo_id,
            revision=revision,
            token=os.getenv("HF_HUB_KEY"),
        )

    # Handle local model_only_checkpoint_path
    if model_only_path is not None:
        if not os.path.isfile(model_only_path):
            raise ValueError(
                f"Model-only checkpoint path does not exist: {model_only_path}"
            )
        return model_only_path

    return None
