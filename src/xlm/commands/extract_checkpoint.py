from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import hydra
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from xlm.utils.checkpoint_paths import is_consolidatable_lightning_sharded_dir
from xlm.utils.consolidate_model_checkpoint import consolidate_model_checkpoint
from xlm.utils.model_loading import load_model_for_inference
from xlm.utils.rank_zero import RankedLogger
from xlm.utils.rich_utils import print_config_tree

logger = RankedLogger(__name__, rank_zero_only=True)


def _normalize_max_shard(val: Any) -> Union[str, int, None]:
    if val is None:
        return None
    try:
        if OmegaConf.is_missing(val):
            return None
    except (ValueError, TypeError, AttributeError):
        pass
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in ("none", "null", "~", ""):
            return None
        return s
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val


def extract_checkpoint(cfg: DictConfig) -> None:
    """
    Args:
        cfg: Hydra config
          cfg.post_training.checkpoint_path: path to existing checkpoint
          cfg.post_training.apply_ema: whether to apply EMA weights to the model weights when loading from checkpoint.
          cfg.post_training.model_state_dict_path (optional): path to save model weights. If provided, the model weights will be saved to the local file system.
          cfg.post_training.repo_id (optional): HuggingFace Hub repository ID. If provided, the model weights will be pushed to the hub.
          cfg.post_training.commit_message: commit message for HuggingFace Hub
          cfg.post_training.max_shard_size (optional): when ``checkpoint_path`` is an FSDP sharded directory,
            optional HF shard size (e.g. ``\"5GB\"``) for multi-file ``.safetensors`` output; ``model_state_dict_path`` must then be a directory.

    Returns:
        None
    """
    if cfg.post_training is None:
        raise ValueError("cfg.post_training is not set. Can't extract checkpoint.")
    if cfg.post_training.get("model_state_dict_path", None) is None and cfg.post_training.get("repo_id", None) is None:
        raise ValueError("Both cfg.post_training.model_state_dict_path and cfg.post_training.repo_id are not set. Don't know what to do with the model weights.")
    # region: common setup shared by eval and generate
    print_config_tree(cfg, resolve=True, save_to_file=False)
    if cfg.get("seed"):
        logger.info(f"Seed everything with seed {cfg.seed}")
        seed_everything(cfg.seed)

    # Always create the global components first.
    global_components: Dict[str, Any] = hydra.utils.instantiate(
        cfg.global_components
    )
    OmegaConf.clear_resolver("global_components")
    OmegaConf.register_new_resolver(
        "global_components", lambda x: global_components[x]
    )
    # instantiate the datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    tokenizer = datamodule.tokenizer

    # update the omegaconf resolvers
    OmegaConf.clear_resolver("tokenizer")
    OmegaConf.register_new_resolver(
        "tokenizer", lambda x: getattr(tokenizer, x)
    )
    OmegaConf.clear_resolver("datamodule")
    OmegaConf.register_new_resolver(
        "datamodule", lambda x: getattr(datamodule, x)
    )
    datamodule.no_trainer_mode = True
    datamodule.prepare_data()
    datamodule.setup("predict")
    # endregion

    # Get checkpoint path and extraction settings
    checkpoint_path = cfg.post_training.checkpoint_path
    if checkpoint_path is None:
        raise ValueError("cfg.post_training.checkpoint_path is not set. Can't extract checkpoint.")

    apply_ema = cfg.post_training.get("apply_ema", True)
    model_state_dict_path = cfg.post_training.get("model_state_dict_path", None)
    repo_id = cfg.post_training.get("repo_id", None)

    ckpt_path = Path(checkpoint_path).expanduser()

    if ckpt_path.is_file():
        # Load Harness from checkpoint with optional EMA (single-file Lightning ckpt)
        logger.info(f"Loading Harness from checkpoint: {checkpoint_path}")
        harness_cls = hydra.utils.get_class(cfg.lightning_module._target_)
        harness = harness_cls.from_checkpoint(
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            tokenizer=tokenizer,
            datamodule=datamodule,
            apply_ema=apply_ema,
        )

        # Save model weights to local file if path is provided
        if model_state_dict_path is not None:
            logger.info(f"Saving model weights to: {model_state_dict_path}")
            harness.save_model_weights(model_state_dict_path, overwrite=True)

        # Push to hub if repo_id is provided
        if repo_id is not None:
            commit_message = cfg.post_training.get("commit_message", "Upload model weights")
            logger.info(f"Pushing model to hub: {repo_id}")
            harness.push_to_hub(repo_id=repo_id, commit_message=commit_message)
        return

    if is_consolidatable_lightning_sharded_dir(ckpt_path):
        if apply_ema:
            raise ValueError(
                "apply_ema=True is not supported for FSDP sharded checkpoints. "
                "Set apply_ema=False, or train with state_dict_type=full and use a single-file .ckpt."
            )
        max_shard_raw = _normalize_max_shard(
            cfg.post_training.get("max_shard_size", None)
        )
        max_shard_size: Union[str, int, None]
        if max_shard_raw is None:
            max_shard_size = None
        elif isinstance(max_shard_raw, int):
            max_shard_size = max_shard_raw
        else:
            max_shard_size = str(max_shard_raw)

        if model_state_dict_path is not None:
            intermediate = Path(model_state_dict_path).expanduser()
        else:
            tmp = Path(tempfile.mkdtemp(prefix="extract_ckpt_"))
            if max_shard_size is not None:
                intermediate = tmp / "model_export"
                intermediate.mkdir(parents=True, exist_ok=True)
            else:
                intermediate = tmp / "model.safetensors"

        logger.info(f"Consolidating FSDP sharded checkpoint to model-only safetensors: {ckpt_path}")
        consolidated = consolidate_model_checkpoint(
            ckpt_path,
            intermediate,
            max_shard_size=max_shard_size,
        )

        if repo_id is not None:
            OmegaConf.update(
                cfg,
                "model_only_checkpoint_path",
                str(consolidated),
                force_add=True,
            )
            commit_message = cfg.post_training.get("commit_message")
            if commit_message is None:
                commit_message = f"Upload model weights from {ckpt_path.name}"
            logger.info(f"Loading Harness for Hub push from: {consolidated}")
            harness, _ = load_model_for_inference(
                cfg,
                datamodule,
                tokenizer,
                config_prefix="",
                manual_ema_restore=False,
                move_to_device="cpu",
                set_eval_mode=True,
                enable_hub_support=False,
                allow_random_init=False,
            )
            logger.info(f"Pushing model to hub: {repo_id}")
            harness.push_to_hub(repo_id=repo_id, commit_message=commit_message)
        else:
            logger.info("Wrote consolidated model-only safetensors to %s", consolidated)
        return

    raise ValueError(
        f"Unrecognized checkpoint layout (expected a single .ckpt file or an FSDP sharded "
        f"directory with *.distcp and meta.pt): {ckpt_path}"
    )