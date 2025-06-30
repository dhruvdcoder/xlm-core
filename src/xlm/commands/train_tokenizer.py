# %%
# change dir to the root of the project
# create the notebook inside the commands directory
import logging
import os

# region: Import necessary modules
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.fabric.utilities.seed import seed_everything
from xlm import utils
from xlm.utils.debug import set_flags

# endregion


# Hydra configuration parameters
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(Path("../../../configs") / "train"),
    "config_name": "train_tokenizer.yaml",
}

# region: Initialize the logger
logger = logging.getLogger(__name__)

# other global constants and functions go here
# ...
# endregion


def train_tokenizer(cfg: DictConfig):
    # Main processing logic for the script (move it inside `foo` function)
    if cfg.get("seed"):
        logger.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed)

    # load data
    train_data = hydra.utils.instantiate(cfg.train_data)

    def batch_iterator(data, batch_size=1000):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]["text"]

    # %%
    # create tokenizer trainer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer_components.tokenizer)
    tokenizer.normalizer = hydra.utils.instantiate(
        cfg.tokenizer_components.normalizer
    )
    tokenizer.pre_tokenizer = hydra.utils.instantiate(
        cfg.tokenizer_components.pre_tokenizer
    )
    tokenizer.decoder = hydra.utils.instantiate(
        cfg.tokenizer_components.decoder
    )
    trainer = hydra.utils.instantiate(cfg.tokenizer_components.trainer)
    # train tokenizer
    tokenizer.train_from_iterator(
        batch_iterator(train_data, batch_size=cfg.batch_size),
        trainer=trainer,
        length=len(train_data),
    )

    # save tokenizer
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "tokenizer.json"
    logger.info(f"Saving tokenizer to {output_file}")
    tokenizer.save(str(output_file))


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    set_flags(cfg)
    try:
        train_tokenizer(cfg)
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    main()
