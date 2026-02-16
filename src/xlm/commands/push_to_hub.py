import os

from xlm.model import Model
from xlm.utils.debug import set_flags
from xlm.external_models import setup_external_models

if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "."
os.environ["HYDRA_FULL_ERROR"] = "1"

# region: Import necessary modules
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from xlm.utils import omegaconf_resolvers
import dotenv
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

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
from xlm.utils.rich_utils import print_config_tree
from lightning import seed_everything
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class PushToHubConfig:
    """Configuration for pushing model to Hugging Face Hub."""

    repo_id: str
    commit_message: str = "Programmatic push to hub"


# Register the config schema
cs = ConfigStore.instance()
cs.store(name="default", group="hub", node=PushToHubConfig)


def instantiate_model(
    cfg: DictConfig,
    datamodule: Any,
    tokenizer: Any,
) -> Harness:
    """Instantiate a model from checkpoint or config.

    Args:
        cfg: Hydra config
        datamodule: Datamodule instance
        tokenizer: Tokenizer instance

    Returns:
        Harness: The instantiated model
    """
    torch.set_float32_matmul_precision("medium")

    # Check for hub checkpoint path
    hub_ckpt_path = None
    if cfg.get("hub_checkpoint_path", None) is not None:
        if not os.path.isfile(cfg.hub_checkpoint_path):
            raise ValueError(
                f"The hub checkpoint path {cfg.hub_checkpoint_path} does not exist."
            )
        hub_ckpt_path = cfg.hub_checkpoint_path

    # Check for model only checkpoint path
    model_only_ckpt_path = None
    if cfg.get("model_only_checkpoint_path", None) is not None:
        if hub_ckpt_path is not None:
            logger.error(
                "hub_checkpoint_path and model_only_checkpoint_path cannot both be provided. "
                "We will use hub_checkpoint_path for the model weights as well."
            )
        else:
            if not os.path.isfile(cfg.model_only_checkpoint_path):
                raise ValueError(
                    f"The model only checkpoint path {cfg.model_only_checkpoint_path} does not exist."
                )
            model_only_ckpt_path = cfg.model_only_checkpoint_path

    # Validate that at least one checkpoint path is provided
    if hub_ckpt_path is None and model_only_ckpt_path is None:
        raise ValueError(
            "At least one checkpoint path must be provided for push_to_hub. "
            "Please provide either `hub_checkpoint_path` or `model_only_checkpoint_path`."
        )

    # Load from hub checkpoint if provided
    if hub_ckpt_path is not None:
        module_cls = hydra.utils.get_class(cfg.lightning_module._target_)
        lightning_module = module_cls.load_from_checkpoint(
            checkpoint_path=hub_ckpt_path,
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

    # Load model only checkpoint if provided
    if model_only_ckpt_path is not None:
        message = lightning_module.model.load_state_dict(
            torch.load(model_only_ckpt_path, map_location="cpu")
        )
        logger.warning(
            f"Loading weights for `model` from a pretrained model at {model_only_ckpt_path} before pushing to hub"
        )
        logger.warning(message)

    return lightning_module


def push_to_hub(cfg: DictConfig):
    # We instantiate everything to re-create the same setup we used for training
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
    model: Model = lightning_module.model
    # Convert hub config to dictionary and push the model to hub
    hub_config: Dict[str, Any] = OmegaConf.to_container(cfg.hub, resolve=True)
    if not hub_config.get("commit_message"):
        commit_message = "Paths:"
        commit_message += (
            f"\n- hub_checkpoint_path: {cfg.get('hub_checkpoint_path', None)}"
        )
        commit_message += f"\n- model_only_checkpoint_path: {cfg.get('model_only_checkpoint_path', None)}"
        hub_config["commit_message"] = commit_message

    model.push_to_hub(**hub_config, token=os.getenv("HF_HUB_KEY"))


def replace_model(cfg: DictConfig) -> DictConfig:
    if "hub_model" in cfg:
        cfg.model = cfg.hub_model
        del cfg.hub_model
        if "generation" in cfg:
            del cfg.generation
    return cfg


# Hydra configuration parameters
# Use absolute path to configs directory within the package
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(
        (
            Path(__file__).parent.parent / "configs" / "lightning_train"
        ).resolve()
    ),
    "config_name": "config.yaml",
}

from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.config_search_path import ConfigSearchPath

hydra_plugins = Plugins.instance()
class HydraSearchPathPlugin(SearchPathPlugin):
        def manipulate_search_path(
            self, search_path: ConfigSearchPath
        ) -> None:
            search_path.append("file", str(Path(__file__).parent.parent / "configs/common"))

hydra_plugins.register(HydraSearchPathPlugin)

external_model_dirs = setup_external_models()
# Register our SearchPathPlugin manually with Hydra
if external_model_dirs:
    class ExternalModelsSearchPathPlugin(SearchPathPlugin):
        def manipulate_search_path(
            self, search_path: ConfigSearchPath
        ) -> None:
            for model_dir in external_model_dirs:
                config_dir = model_dir / "configs"
                if config_dir.exists():
                    search_path.append("file", str(config_dir))

    # Register the plugin
    hydra_plugins.register(ExternalModelsSearchPathPlugin)


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    """Main function for Push to Hub."""
    # global flags
    set_flags(cfg)
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

    # Run the push to hub
    push_to_hub(cfg)


if __name__ == "__main__":
    main()
