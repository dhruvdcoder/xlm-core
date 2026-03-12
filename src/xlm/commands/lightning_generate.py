import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, Optional, cast

import torch

from xlm.harness import Harness
from xlm.utils.model_loading import load_model_for_inference
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
    """Instantiate and load a model for generation.

    Supports loading from full checkpoints, model-only checkpoints, or HF Hub.

    Args:
        cfg: Hydra config
        datamodule: Datamodule
        tokenizer: Tokenizer

    Returns:
        Loaded Harness model ready for generation
    """
    module, _ = load_model_for_inference(
        cfg,
        datamodule,
        tokenizer,
        config_prefix="generation",
        manual_ema_restore=True,
        move_to_device="cuda",
        set_eval_mode=True,
        enable_hub_support=True,
        allow_random_init=False,
    )
    return module


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

    # add generation config if not present
    if "generation" not in cfg:
        OmegaConf.update(cfg, "generation", {}, force_add=True)
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
