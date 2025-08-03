# %%
# change dir to the root of the project
# create the notebook inside the commands directory
import os

from xlm.utils.debug import set_flags
from xlm.utils.slurm import print_slurm_info

if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "."
os.environ["HYDRA_FULL_ERROR"] = "1"

# region: Import necessary modules
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from xlm.utils import omegaconf_resolvers
import dotenv

# endregion

# region: other global constants and functions
dotenv.load_dotenv(
    override=True
)  # set env variables from .env file, override=True is important
found_secretes = dotenv.load_dotenv(".secrets.env", override=True)
if not found_secretes:
    print("Warning: .secrets.env not found")
# endregion

from typing import Any, Dict, cast

import torch

from xlm.harness import Harness
from xlm.utils.rich_utils import print_config_tree
from lightning import seed_everything
from xlm.utils.rank_zero import RankedLogger
from .cli_demo import replace_model

logger = RankedLogger(__name__, rank_zero_only=True)


def instantiate_model(
    cfg: DictConfig,
    datamodule: Any,
    tokenizer: Any,
) -> Harness:
    """Instantiate a model from checkpoint or config.

    Supports two modes:
        1. Load a model from full training checkpoint using `lightning_module.load_from_checkpoint(cfg.generation.ckpt_path)`
        2. Load a model from model only checkpoint using `lightning_module.model.load_state_dict(torch.load(cfg.generation.model_only_checkpoint_path))`

    Args:
        cfg: Hydra config
        datamodule: Datamodule instance
        tokenizer: Tokenizer instance

    Returns:
        Harness: The instantiated model
    """
    generation_ckpt_path = None
    if "generation" in cfg:
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
    if (
        "generation" in cfg
        and cfg.generation.get("model_only_checkpoint_path", None) is not None
    ):
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
    """Generate text using the CLI demo interface."""
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

    # instantiate the model
    lightning_module = instantiate_model(cfg, datamodule, tokenizer)

    # get user input and predict in a loop
    with torch.inference_mode():
        while True:
            user_input = input("Enter your prompt (or 'exit' to quit): ")
            if user_input == "exit":
                break
            user_input_list = [user_input]
            preds = lightning_module.predictor.generate(user_input_list)
            print(preds[0])


# Hydra configuration parameters for CLI demo
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(Path("../../../configs") / "lightning_train"),
    "config_name": "config.yaml",
}


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    """Main function for CLI demo."""
    # global flags
    set_flags(cfg)
    # make early changes to the config
    cfg = replace_model(cfg)
    # Register resolvers
    omegaconf_resolvers.register_resolvers()
    OmegaConf.register_new_resolver(
        "datamodule", lambda attr: "${datamodule:" + str(attr) + "}"
    )
    OmegaConf.register_new_resolver(
        "tokenizer", lambda attr: "${tokenizer:" + str(attr) + "}"
    )
    OmegaConf.register_new_resolver(
        "lightning_module",
        lambda attr: "${lightning_module:" + str(attr) + "}",
    )
    OmegaConf.register_new_resolver(
        "global_components",
        lambda attr: "${global_components:" + str(attr) + "}",
    )

    # Run the generation
    generate(cfg)


if __name__ == "__main__":
    main()
