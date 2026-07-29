# %%
# Interactive CLI demo for prompt → generation.
import os

from xlm.utils.debug import set_flags

if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "."
os.environ["HYDRA_FULL_ERROR"] = "1"

# region: Import necessary modules
from pathlib import Path
from typing import Any, Dict

import dotenv
import hydra
import torch
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from xlm.external_models import setup_external_models
from xlm.harness import Harness
from xlm.utils import omegaconf_resolvers
from xlm.utils.model_loading import load_model_for_inference
from xlm.utils.rank_zero import RankedLogger
from xlm.utils.rich_utils import print_config_tree

# endregion

# region: other global constants and functions
dotenv.load_dotenv(
    override=True
)  # set env variables from .env file, override=True is important
found_secrets = dotenv.load_dotenv(".secrets.env", override=True)
if not found_secrets:
    print("Warning: .secrets.env not found")
# endregion

logger = RankedLogger(__name__, rank_zero_only=True)


def replace_model(cfg: DictConfig) -> DictConfig:
    """Legacy +hub/checkpoint=* path: swap model to hub from_pretrained target."""
    if "hub_model" in cfg:
        cfg.model = cfg.hub_model
        del cfg.hub_model
        if "generation" in cfg:
            del cfg.generation
    return cfg


def instantiate_model(
    cfg: DictConfig,
    datamodule: Any,
    tokenizer: Any,
) -> Harness:
    """Instantiate a model from checkpoint / Hub for interactive CLI demo."""
    # Hub weights via +hub.repo_id / +hub.revision (same as job_type=generate).
    # Legacy +hub/checkpoint=* still works via replace_model → from_pretrained;
    # in that case allow_random_init is needed because weights are already in the
    # model object and there is no generation.* checkpoint path.
    use_hub_model = "hub_model" in cfg or (
        cfg.get("model") is not None
        and "from_pretrained" in str(cfg.model.get("_target_", ""))
    )
    module, _ = load_model_for_inference(
        cfg,
        datamodule,
        tokenizer,
        config_prefix="generation",
        manual_ema_restore=False,
        move_to_device="cuda",
        set_eval_mode=True,
        enable_hub_support=True,
        allow_random_init=use_hub_model,
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
    "config_path": str(
        (
            Path(__file__).parent.parent / "configs" / "lightning_train"
        ).resolve()
    ),
    "config_name": "config.yaml",
}

hydra_plugins = Plugins.instance()


class HydraCommonSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            "file", str(Path(__file__).parent.parent / "configs/common")
        )


hydra_plugins.register(HydraCommonSearchPathPlugin)

external_model_dirs = setup_external_models()
if external_model_dirs:

    class ExternalModelsSearchPathPlugin(SearchPathPlugin):
        def manipulate_search_path(
            self, search_path: ConfigSearchPath
        ) -> None:
            for model_dir in external_model_dirs:
                config_dir = model_dir / "configs"
                if config_dir.exists():
                    search_path.append("file", str(config_dir))

    hydra_plugins.register(ExternalModelsSearchPathPlugin)


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
