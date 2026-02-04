import os
from typing import Any, Dict, List, Optional, cast

import torch

from xlm.harness import Harness


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
    """
    Instantiate and load a model for evaluation.

    Supports two modes:
        1. Load a model from full training checkpoint using `lightning_module.load_from_checkpoint(cfg.eval.checkpoint_path)`
        2. Load a model from model only checkpoint using `lightning_module.model.load_state_dict(torch.load(cfg.eval.model_only_checkpoint_path))`

    Args:
        cfg: Hydra config
        datamodule: Datamodule
        tokenizer: Tokenizer

    Returns:
        Tuple of (lightning_module, ckpt_path) where ckpt_path is the full checkpoint path
        (or None if using model-only checkpoint)
    """
    torch.set_float32_matmul_precision("medium")

    # Determine checkpoint path
    ckpt_path = cfg.get("eval", {}).get("checkpoint_path", None)
    # try to use "best.ckpt" if no checkpoint path is provided
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.checkpointing_dir, "best.ckpt")
        if not os.path.isfile(ckpt_path):
            logger.info(f"No checkpoint found at {ckpt_path}")
            ckpt_path = None
    # try using "last.ckpt" if no checkpoint path is provided
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.checkpointing_dir, "last.ckpt")
        if not os.path.isfile(ckpt_path):
            logger.info(f"No checkpoint found at {ckpt_path}")
            ckpt_path = None

    # Check for model-only checkpoint path
    model_only_ckpt_path = cfg.get("eval", {}).get(
        "model_only_checkpoint_path", None
    )

    # Validation: can't have both
    if ckpt_path is not None and model_only_ckpt_path is not None:
        logger.error(
            "eval.model_only_checkpoint_path and eval.checkpoint_path cannot both be provided. "
            "We will use eval.checkpoint_path for the model weights as well."
        )
        model_only_ckpt_path = None

    if ckpt_path is None and model_only_ckpt_path is None:
        raise ValueError("No checkpoint found")

    module_cls = hydra.utils.get_class(cfg.lightning_module._target_)

    # Load from full checkpoint or instantiate new model
    if ckpt_path is not None:
        logger.info(f"Evaluating checkpoint {ckpt_path}")
        # We don't need to manually restore the EMA weights when loading from checkpoint
        # for evaluation because we have callbacks (EMACallback).
        lightning_module = module_cls.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            tokenizer=tokenizer,
            datamodule=datamodule,
            cfg=cfg,  # chance to override the config of the checkpoint
        )
        return_ckpt_path = ckpt_path
    else:
        # Instantiate a new model when using model-only checkpoint
        logger.info(
            "Instantiating new lightning module for model-only checkpoint"
        )
        lightning_module = hydra.utils.instantiate(
            cfg.lightning_module,
            tokenizer=tokenizer,
            datamodule=datamodule,
            cfg=cfg,
            _recursive_=False,
        )
        return_ckpt_path = None

    lightning_module = cast(Harness, lightning_module)

    # Load model-only checkpoint if provided
    if model_only_ckpt_path is not None:
        if not os.path.isfile(model_only_ckpt_path):
            raise ValueError(
                f"The model only checkpoint path {model_only_ckpt_path} does not exist."
            )
        logger.info(f"Loading model weights from {model_only_ckpt_path}")
        message = lightning_module.model.load_state_dict(
            torch.load(model_only_ckpt_path)
        )
        logger.warning(
            f"Loading weights for `model` from a pretrained model at {model_only_ckpt_path} before evaluation. "
            "Make sure that the model weights were saved with EMA applied if needed."
        )
        logger.warning(message)
        # Ensure we don't pass a checkpoint path to trainer when using model-only weights
        return_ckpt_path = None

    return lightning_module, return_ckpt_path


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
