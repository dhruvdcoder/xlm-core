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
found_secrets = dotenv.load_dotenv(".secrets.env", override=True)
if not found_secrets:
    print("Warning: .secrets.env not found")
# endregion

from typing import Any, Dict, cast

import torch

from xlm.harness import Harness
from xlm.utils.model_loading import load_model_for_inference
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
    """Instantiate a model from checkpoint for interactive CLI demo.

    Args:
        cfg: Hydra config
        datamodule: Datamodule instance
        tokenizer: Tokenizer instance

    Returns:
        Harness: The instantiated model ready for demo
    """
    module, _ = load_model_for_inference(
        cfg,
        datamodule,
        tokenizer,
        config_prefix="generation",
        manual_ema_restore=False,
        move_to_device="cuda",
        set_eval_mode=True,
        enable_hub_support=False,
        allow_random_init=False,
    )
    return module


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
