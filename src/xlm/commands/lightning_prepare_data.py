# %%
# change dir to the root of the project
# create the notebook inside the commands directory
import os
from typing import Any, Dict


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


def prepare_data(cfg: DictConfig):
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
    # run prepare_data and exit
    datamodule.prepare_data()
