# %%
# change dir to the root of the project
# create the notebook inside the commands directory
import os

import torch

if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "."
os.environ["HYDRA_FULL_ERROR"] = "1"

# region: Import necessary modules
import contextlib
from typing import Any, Dict, List
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything
from lightning.pytorch.loggers import Logger
from lightning import Callback
from transformers.modeling_utils import no_init_weights
from xlm.utils.hf_hub import (
    download_model_weights,
    load_model_weights_into_model,
    repo_id_from_hf_path,
)
from xlm.utils.model_loading import _default_dtype_ctx, _resolve_init_dtype
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
        if cfg.get("callbacks_creator", None) is None:
            for _, cb_conf in cfg.callbacks.items():
                if cb_conf is not None and "_target_" in cb_conf:
                    logger.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))
        else:
            creator = hydra.utils.instantiate(
                cfg.callbacks_creator, cfg=cfg, _recursive_=False
            )
            callbacks = creator(cfg)

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

    if model_only_ckpt_path is None and ckpt_path is None:
        hub_repo_id = OmegaConf.select(cfg, "hub.repo_id", default=None)
        if hub_repo_id is not None:
            repo_id = repo_id_from_hf_path(hub_repo_id) or hub_repo_id
            revision = OmegaConf.select(cfg, "hub.revision", default="main")
            logger.info(
                "Downloading model weights from Hugging Face Hub for training: "
                f"{repo_id} (revision={revision})"
            )
            model_only_ckpt_path = download_model_weights(
                repo_id=repo_id,
                revision=revision,
                token=os.getenv("HF_HUB_KEY"),
            )
    elif ckpt_path is not None and OmegaConf.select(
        cfg, "hub.repo_id", default=None
    ) is not None:
        logger.warning(
            "Ignoring hub.repo_id because a Lightning resume checkpoint is set."
        )

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )
    # REMOVE this. The number of samples are set from the datamodule.
    ## set the number of samples for the sample generator
    # num_samples = int(cfg.generative_perplexity.num_samples) // (
    #    trainer.num_devices * trainer.num_nodes
    # )
    # if (
    #    int(cfg.generative_perplexity.num_samples)
    #    % (trainer.num_devices * trainer.num_nodes)
    #    != 0
    # ):
    #    logger.warning(
    #        f"The number of samples for generative perplexity ({cfg.generative_perplexity.num_samples}) is not divisible by the (number of devices * number of nodes)=({trainer.num_devices * trainer.num_nodes}). "
    #    )
    # logger.info(
    #    f"Setting the total number of samples for generative perplexity to {num_samples*trainer.num_devices*trainer.num_nodes}"
    # )
    # logger.info(
    #    f"Setting per device number of samples for generative perplexity to {num_samples}"
    # )
    # instantiate the model
    torch.set_float32_matmul_precision("medium")
    will_load_weights = ckpt_path is not None or model_only_ckpt_path is not None
    skip_init = (
        model_only_ckpt_path is not None and ckpt_path is None
    ) or (
        cfg.get("skip_init_weights", False) and will_load_weights
    )
    if skip_init:
        logger.info(
            "Skipping random weight init (pretrained or Lightning checkpoint will "
            "supply weights; lowers peak CPU RAM during module construction)."
        )
    skip_ctx = no_init_weights() if skip_init else contextlib.nullcontext()
    init_dtype = _resolve_init_dtype(cfg)
    if init_dtype is not None:
        logger.info(f"Instantiating lightning module with init_dtype={init_dtype}")
    with _default_dtype_ctx(init_dtype), skip_ctx:
        lightning_module = hydra.utils.instantiate(
            cfg.lightning_module,
            cfg,
            tokenizer=tokenizer,
            datamodule=datamodule,
            _recursive_=False,
        )
    if model_only_ckpt_path is not None:
        load_target = getattr(
            lightning_module.model, "dream", lightning_module.model
        )
        logger.info(
            f"Loading pretrained weights into {type(load_target).__name__} from "
            f"{model_only_ckpt_path}"
        )
        load_model_weights_into_model(
            load_target,
            model_only_ckpt_path,
            map_location="cpu",
            strict=True,
            weights_only=True,
        )

    # init_dtype lowers peak RAM during construction, but training uses float32 batch
    # tensors and optimizers unless the Trainer runs full bf16 — mixed dtypes then
    # break matmuls. Cast parameters back to float32 for the fit loop.
    if init_dtype is not None and init_dtype != torch.float32:
        logger.info(
            "Casting module parameters to float32 for training "
            f"(construction used init_dtype={init_dtype})."
        )
        lightning_module.float()

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
    ## separate test run
    if cfg.job_type == "test":  # if separate test run
        test_ckpt = ckpt_path or "last"
    try:
        logger.info(f"Testing with checkpoint: {test_ckpt or 'last'}")
        trainer.test(
            model=lightning_module,
            datamodule=datamodule,
            ckpt_path=test_ckpt,
        )
    except Exception as e:
        logger.error(f"Could not run test: {e}")
