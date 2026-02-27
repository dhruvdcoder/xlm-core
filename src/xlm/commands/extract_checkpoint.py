from typing import Any, Dict
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything
from xlm.utils.rank_zero import RankedLogger
from xlm.utils.rich_utils import print_config_tree

logger = RankedLogger(__name__, rank_zero_only=True)

def extract_checkpoint(cfg: DictConfig) -> None:
    """
    Args:
        cfg: Hydra config
          cfg.post_training.checkpoint_path: path to existing checkpoint
          cfg.post_training.apply_ema: whether to apply EMA weights to the model weights when loading from checkpoint.
          cfg.post_training.model_state_dict_path (optional): path to save model weights. If provided, the model weights will be saved to the local file system.
          cfg.post_training.repo_id (optional): HuggingFace Hub repository ID. If provided, the model weights will be pushed to the hub.
          cfg.post_training.commit_message: commit message for HuggingFace Hub

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

    # Load Harness from checkpoint with optional EMA
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