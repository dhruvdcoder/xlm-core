from curses import savetty
import logging
from pathlib import Path
from typing import Sequence, Union

import omegaconf
import torch

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from .os import is_notebook
from .rank_zero import rank_zero_only
from rich.prompt import Prompt

logger = logging.getLogger(__name__)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = [],
    resolve: bool = True,
    save_to_file: Union[bool, str] = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines what top level fields to print and in what order.
        resolve (bool, optional): Whether to resolve reference fields of
            DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra
            output folder.
    """
    # Saving as yaml file is better for readability
    yaml = omegaconf.OmegaConf.to_yaml(cfg, resolve=resolve)
    # save to file
    if save_to_file:
        if isinstance(save_to_file, str):
            dir_ = Path(save_to_file)
        else:
            dir_ = Path(cfg.paths.output_dir)

        dir_.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving config tree to {dir_ / 'config_tree.yaml'}")
        with open(dir_ / "config_tree.yaml", "w") as file:
            file.write(yaml)

    # detect notebook and disable rich
    if is_notebook():
        print(yaml)
        return

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue

    for field in print_order:
        (
            queue.append(field)
            if field in cfg
            else logger.warning(
                f"Field '{field}' not found in config. Skipping '{field}' config printing..."
            )
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    # Create the output directory if it doesn't exist
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    if save_to_file:
        logger.info(
            f"Saving config tree to {Path(cfg.paths.output_dir) / 'config_tree.txt'}"
        )
        with open(Path(cfg.paths.output_dir) / "config_tree.txt", "w") as file:
            console = rich.console.Console(file=file)
            console.print(tree)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in
    config."""
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        logger.warning(
            "No tags provided in config. Prompting user to input tags..."
        )
        tags = Prompt.ask(
            "Enter a list of comma separated tags", default="dev"
        )
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        logger.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)
