import os

import torch


if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "."
os.environ["HYDRA_FULL_ERROR"] = "1"

# region: Import necessary modules
from typing import Any, Dict, List
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything
from lightning.pytorch.loggers import Logger
from lightning import Callback
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# endregion


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
    if ckpt_path is None:
        raise ValueError("No checkpoint found")
    logger.info(f"Evaluating checkpoint {ckpt_path}")

    # cls = hydra.utils.get_class(cfg.lightning_module._target_)
    torch.set_float32_matmul_precision("medium")
    module_cls = hydra.utils.get_class(cfg.lightning_module._target_)
    # We don't need to manually restore the EMA weights when loading from checkpoint for evaluation because we have callbacks.
    lightning_module = module_cls.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        tokenizer=tokenizer,
        datamodule=datamodule,
        cfg=cfg,  # chance to override the config of the checkpoint
    )
    split = cfg.get("eval", {}).get("split", "validation")
    if split == "validation":
        trainer.validate(
            model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path
        )
    elif split == "test":
        trainer.test(
            model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path
        )
    else:
        raise ValueError(f"Invalid split: {cfg.eval.split}")
