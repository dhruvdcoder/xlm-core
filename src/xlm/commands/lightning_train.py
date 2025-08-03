# %%
# change dir to the root of the project
# create the notebook inside the commands directory
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


def train(cfg: DictConfig):
    print("num_gpus: ", torch.cuda.device_count())
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
                if "MyWandbLogger" in lg_conf._target_:
                    loggers.append(
                        hydra.utils.instantiate(
                            lg_conf, hydra_cfg=cfg, _recursive_=False
                        )
                    )
                else:
                    loggers.append(hydra.utils.instantiate(lg_conf))

    # Resume from checkpoint and automatic checkpoint pickup
    ckpt_path = None
    if cfg.resume_from_checkpoint:
        # determine the checkpoint path
        if cfg.resume_checkpoint_path is not None:
            if os.path.isfile(cfg.resume_checkpoint_path):
                ckpt_path = cfg.resume_checkpoint_path
            else:
                raise ValueError(
                    f"The checkpoint path {cfg.resume_checkpoint_path} is not a file."
                )
        else:
            # look for the "last.ckpt" or "on_exception.ckpt" checkpoint in the checkpointing_dir
            ckpt_path = os.path.join(
                cfg.checkpointing_dir, "on_exception.ckpt"
            )
            if not os.path.isfile(ckpt_path):
                ckpt_path = os.path.join(cfg.checkpointing_dir, "last.ckpt")
            if not os.path.isfile(ckpt_path):
                ckpt_path = None
    # check if we have model only checkpoint
    model_only_ckpt_path = None
    if cfg.get("model_only_checkpoint_path", None) is not None:
        if ckpt_path is not None:
            logger.error(
                "model_only_checkpoint_path and resume_from_checkpoint cannot both be provided."
                " We will use the resume_from_checkpoint path "
                f"{ckpt_path} for the model weights as well."
            )
        else:
            if not os.path.isfile(cfg.model_only_checkpoint_path):
                raise ValueError(
                    f"The model only checkpoint path {cfg.model_only_checkpoint_path} does not exist."
                )
            model_only_ckpt_path = cfg.model_only_checkpoint_path

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )
    # set the number of samples for the sample generator
    num_samples = int(cfg.generative_perplexity.num_samples) // (
        trainer.num_devices * trainer.num_nodes
    )
    if (
        int(cfg.generative_perplexity.num_samples)
        % (trainer.num_devices * trainer.num_nodes)
        != 0
    ):
        logger.warning(
            f"The number of samples for generative perplexity ({cfg.generative_perplexity.num_samples}) is not divisible by the (number of devices * number of nodes)=({trainer.num_devices * trainer.num_nodes}). "
        )
    logger.info(
        f"Setting the total number of samples for generative perplexity to {num_samples*trainer.num_devices*trainer.num_nodes}"
    )
    logger.info(
        f"Setting per device number of samples for generative perplexity to {num_samples}"
    )
    # instantiate the model
    torch.set_float32_matmul_precision("medium")
    lightning_module = hydra.utils.instantiate(
        cfg.lightning_module,
        cfg,
        tokenizer=tokenizer,
        datamodule=datamodule,
        _recursive_=False,
    )
    if model_only_ckpt_path is not None:
        message = lightning_module.model.load_state_dict(
            torch.load(model_only_ckpt_path)
        )
        logger.warning(
            "Loading weights for `model` from a pretrained model at "
            f"{model_only_ckpt_path} before call to `trainer.fit` => before `.setup()` and `.configure_model()`"
        )
        logger.warning(message)

    # train
    if cfg.job_type == "train":
        logger.info("Starting training...")
        trainer.fit(
            model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path
        )
    # region: test
    # Run the test in a separate script if possible using a single device.
    # See https://github.com/Lightning-AI/pytorch-lightning/discussions/12856
    ## test after training
    test_ckpt = None
    testing_right_after_training = (
        cfg.job_type == "train"
        and cfg.datamodule.get("test_dataloader_kwargs") is not None
    )
    if testing_right_after_training:
        test_ckpt = None
    ## separate test run
    separate_test_run = cfg.job_type == "test"
    if separate_test_run:
        if cfg.datamodule.get("test_dataloader_kwargs") is None:
            raise ValueError(
                "test_dataloader_kwargs is not provided in the datamodule."
                " Needed for separate test run."
            )
        test_ckpt = ckpt_path or "last"
    logger.info(f"Testing with checkpoint: {test_ckpt or 'last'}")
    if testing_right_after_training or separate_test_run:
        trainer.test(
            model=lightning_module,
            datamodule=datamodule,
            ckpt_path=test_ckpt,
        )

    # endregion: test

    # TODO (training script): Print the best checkpoint

    # TODO (training script): Close the loggers like wandb, etc.

    # TODO (training script): Return the tracked metric value
