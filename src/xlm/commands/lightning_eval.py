import os
from typing import Any, Dict, List, Optional, cast

import torch

from xlm.harness import Harness
from xlm.utils.model_loading import load_model_for_inference


if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "."
os.environ["HYDRA_FULL_ERROR"] = "1"

# region: Import necessary modules
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything
from lightning.pytorch.loggers import Logger
from lightning import Callback
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# endregion


def instantiate_model(
    cfg: DictConfig,
    datamodule: Any,
    tokenizer: Any,
) -> tuple[Harness, Optional[str]]:
    """Instantiate and load a model for evaluation.

    Supports loading from full checkpoints, model-only checkpoints, or HF Hub.
    Returns the checkpoint path for the trainer to use.

    Args:
        cfg: Hydra config
        datamodule: Datamodule
        tokenizer: Tokenizer

    Returns:
        Tuple of (lightning_module, ckpt_path) where ckpt_path is passed to trainer
    """
    return load_model_for_inference(
        cfg,
        datamodule,
        tokenizer,
        config_prefix="eval",
        manual_ema_restore=False,
        move_to_device=None,
        set_eval_mode=False,
        enable_hub_support=True,
        checkpoint_fallback_dir=cfg.checkpointing_dir,
        allow_random_init=False,
    )


def evaluate(cfg: DictConfig):

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

    # instantiate the callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if cb_conf is not None and "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # REMOVE: moved to inside the harness
    # if cfg.generative_perplexity.evaluators is not None:
    #    for (
    #        evaluator_name,
    #        evaluator_conf,
    #    ) in cfg.generative_perplexity.evaluators.items():
    #        evaluator = hydra.utils.instantiate(evaluator_conf)
    #        callbacks.append(GenerativePerplexityCallback(evaluator))

    # instantiate the loggers
    loggers: List[Logger] = []
    if "loggers" in cfg:
        for _, lg_conf in cfg.loggers.items():
            if lg_conf is not None and "_target_" in lg_conf:
                logger.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(hydra.utils.instantiate(lg_conf))

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    # Instantiate the model (supports both full checkpoint and model-only checkpoint)
    lightning_module, ckpt_path = instantiate_model(cfg, datamodule, tokenizer)

    split = cfg.get("eval", {}).get("split", "validation")
    if split == "validation":
        # Pass ckpt_path=None when using model-only checkpoint to avoid Lightning reloading
        trainer.validate(
            model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path
        )
    elif split == "test":
        # Pass ckpt_path=None when using model-only checkpoint to avoid Lightning reloading
        trainer.test(
            model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path
        )
    else:
        raise ValueError(f"Invalid split: {cfg.eval.split}")
