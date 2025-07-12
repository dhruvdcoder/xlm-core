# %%
# change dir to the root of the project
# create the notebook inside the commands directory
import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, Optional, cast

import torch

from xlm.harness import Harness
from xlm.utils.rich_utils import print_config_tree


if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "."
os.environ["HYDRA_FULL_ERROR"] = "1"

# region: Import necessary modules
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# endregion


def get_generation_output_file_path(cfg: DictConfig) -> Optional[Path]:
    generation_output_dir: Optional[Path] = cfg.generation.get(
        "output_dir", None
    )
    generation_output_dir = (
        Path(generation_output_dir)
        if generation_output_dir is not None
        else None
    )
    generation_output_file_name: Optional[str] = cfg.generation.get(
        "output_file_name", None
    )
    if (
        generation_output_file_name is None
        and generation_output_dir is not None
    ):
        raise ValueError(
            "output_file_name is required when output_dir is provided"
        )
    generation_output_file_path: Optional[Path] = None
    if generation_output_file_name is not None:
        assert generation_output_dir is not None
        generation_output_file_path = (
            generation_output_dir / generation_output_file_name
        )
    return generation_output_file_path


def instantiate_model(
    cfg: DictConfig,
    datamodule: Any,
    tokenizer: Any,
) -> Harness:
    """

    Supports two modes:
        1. Load a model from full training checkpoint using `lightning_module.load_from_checkpoint(cfg.generation.ckpt_path)`
        2. Load a model from model only checkpoint using `lightning_module.model.load_state_dict(torch.load(cfg.generation.model_only_checkpoint_path))`

    Args:
        cfg: Hydra config
        generation_ckpt_path: Path to the generation checkpoint
        datamodule: Datamodule
    """
    generation_ckpt_path = cfg.generation.ckpt_path
    torch.set_float32_matmul_precision("medium")
    if generation_ckpt_path is not None:
        module_cls = hydra.utils.get_class(cfg.lightning_module._target_)
        lightning_module = module_cls.load_from_checkpoint(
            checkpoint_path=generation_ckpt_path,
            tokenizer=tokenizer,
            datamodule=datamodule,
            cfg=cfg,  # chance to override the config of the checkpoint
        )
    else:
        lightning_module = hydra.utils.instantiate(
            cfg.lightning_module,
            tokenizer=tokenizer,
            datamodule=datamodule,
            cfg=cfg,
            _recursive_=False,
        )
    lightning_module = cast(Harness, lightning_module)
    lightning_module = lightning_module.to("cuda")
    lightning_module.eval()

    # check if we have model only checkpoint (replicating train functionality)
    model_only_ckpt_path = None
    if cfg.generation.get("model_only_checkpoint_path", None) is not None:
        if generation_ckpt_path is not None:
            logger.error(
                "generation.model_only_checkpoint_path and generation.ckpt_path cannot both be provided. "
                "We will use generation.ckpt_path for the model weights as well."
            )
        else:
            if not os.path.isfile(cfg.model_only_checkpoint_path):
                raise ValueError(
                    f"The model only checkpoint path {cfg.model_only_checkpoint_path} does not exist."
                )
            model_only_ckpt_path = cfg.model_only_checkpoint_path

    if model_only_ckpt_path is not None:
        message = lightning_module.model.load_state_dict(
            torch.load(model_only_ckpt_path)
        )
        logger.warning(
            f"Loading weights for `model` from a pretrained model at {model_only_ckpt_path} before generation"
        )
        logger.warning(message)
    return lightning_module


def generate(cfg: DictConfig):
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

    # instantiate the model
    lightning_module = instantiate_model(cfg, datamodule, tokenizer)

    # output file path
    generation_output_file_path = get_generation_output_file_path(cfg)

    if generation_output_file_path is not None:
        logger.info(
            f"Writing generation output to {generation_output_file_path}"
        )
    predict_dataloaders = datamodule.predict_dataloader()
    dataloader_names = []
    if isinstance(predict_dataloaders, list):
        for i, dl in enumerate(predict_dataloaders):
            dataloader_names.append(datamodule.dataloader_names["predict"][i])
    else:
        dataloader_names.append(datamodule.dataloader_names["predict"][0])
        predict_dataloaders = [predict_dataloaders]

    for dataloader_idx, (dataloader_name, dl) in enumerate(
        zip(dataloader_names, predict_dataloaders)
    ):
        for batch_idx, batch in enumerate(dl):
            # hardcode the move to cuda for now
            batch = {
                k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            preds = lightning_module.predict_step(
                batch, batch_idx, dataloader_idx
            )
            lightning_module._call_log_predictions(
                batch, preds, "predict", dataloader_name, no_trainer_mode=True
            )
