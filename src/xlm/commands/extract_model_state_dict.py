# %%
# change dir to the root of the project
# create the notebook inside the commands directory
import os

import torch


if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "."
os.environ["HYDRA_FULL_ERROR"] = "1"

# region: Import necessary modules
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# endregion


# Hydra configuration parameters


@dataclass
class ExtractModelStateConfig:
    checkpoint_path: Path
    output_path: Path
    add_step_info: bool = False


cs = ConfigStore.instance()
cs.store(name="extract_model_state_dict", node=ExtractModelStateConfig)

_HYDRA_PARAMS = {
    "version_base": "1.3",
    # "config_path": str(Path("../../../configs") / "lightning_train"),
    "config_name": "extract_model_state_dict.yaml",
}


def print_checkpoint_info(full_state_dict: dict) -> None:
    logger.info(f"Checkpoint keys: {full_state_dict.keys()}")
    for k, v in full_state_dict.items():
        if isinstance(v, (int, float, bool, str)):
            logger.info(f"full_state_dict[{k}]: {v}")


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: ExtractModelStateConfig) -> None:
    with open(cfg.checkpoint_path, "rb") as f:
        _full_state_dict = torch.load(f, weights_only=False)
        full_state_dict = _full_state_dict[
            "state_dict"
        ]  # lightning module state_dict
        print_checkpoint_info(_full_state_dict)
        if cfg.add_step_info:
            step = _full_state_dict.get("global_step", None)
            epoch = _full_state_dict.get("epoch", None)
            prefix = f"{epoch=}-{step=}_"
        else:
            prefix = ""
        model_state_dict = {}
        for key in full_state_dict.keys():
            if key.startswith("model."):
                model_state_dict[key[6:]] = full_state_dict[key]
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(cfg.output_path)
    fname = prefix + output_path.name
    output_path = output_path.parent / fname
    torch.save(model_state_dict, output_path)
    logger.info(f"Model state dict saved to {output_path}")


if __name__ == "__main__":
    main()
