"""Custom hydra callbacks for xlm. See https://hydra.cc/docs/experimental/callbacks/ for more information."""

import copy
import logging
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf, flag_override

from hydra.core.global_hydra import GlobalHydra
from hydra.experimental.callback import Callback
from hydra.types import RunMode
from hydra.errors import ConfigCompositionException


def _get_important_default_list_elements(
    defaults_list: List[str],
) -> Dict[str, List[str]]:
    return {
        "datamodule": [
            d for d in defaults_list if d.startswith("datamodule/")
        ],
        "datasets": [d for d in defaults_list if d.startswith("datasets/")],
        "collator": [d for d in defaults_list if d.startswith("collator/")],
        "model": [
            d
            for d in defaults_list
            if d.startswith("model/") and not d.startswith("model_type/")
        ],
        "model_type": [
            d for d in defaults_list if d.startswith("model_type/")
        ],
    }


def _print_important_default_list_elements(
    logger: logging.Logger,
    defaults_list: List[str],
    validate_defaults: bool = True,
) -> None:
    defaults_dict = _get_important_default_list_elements(defaults_list)

    logger.info("Defaults list:")
    for key, _defaults in defaults_dict.items():
        if len(_defaults) >= 1:
            for d in _defaults:
                logger.info(f"  {d}")
        else:
            if validate_defaults:
                raise ConfigCompositionException(
                    f"No defaults found for {key}"
                )
            else:
                logger.warning(f"No defaults found for {key}")


class LogComposeCallback(Callback):
    """Log compose call, result, and debug info"""

    def __init__(self, validate_defaults: bool = True) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.validate_defaults = validate_defaults

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Called in RUN mode before job/application code starts. `config` is composed with overrides.
        Some `hydra.runtime` configs are not populated yet.
        See hydra.core.utils.run_job for more info.
        """
        pass

    def on_compose_config(
        self,
        config: DictConfig,
        config_name: Optional[str],
        overrides: List[str],
    ) -> None:
        gh = GlobalHydra.instance()
        config_loader = gh.config_loader()
        config_dir = "unknown"
        defaults_list = config_loader.compute_defaults_list(
            config_name, overrides, RunMode.RUN
        )
        all_sources = config_loader.get_sources()
        if config_name:  # determine the config directory
            for src in all_sources:
                if src.is_config(config_name):
                    config_dir = src.full_path()
                    break
        original_config = config
        if "hydra" in config:  # remove hydra config from the config
            config = copy.copy(config)
            with flag_override(config, ["struct", "readonly"], [False, False]):
                config.pop("hydra")
        non_hydra_defaults = [
            d.config_path
            for d in defaults_list.defaults
            if not d.package.startswith("hydra")
        ]
        job_type_missing = "job_type" not in overrides

        important_default_list_elements = _get_important_default_list_elements(
            non_hydra_defaults
        )
        original_config.hydra.help.important_default_list_elements = (
            OmegaConf.create(
                important_default_list_elements,
                parent=original_config.hydra.help,
            )
        )
        self.log.info("Selected config groups:")
        for key, _defaults in important_default_list_elements.items():
            if len(_defaults) >= 1:
                for d in _defaults:
                    self.log.info(f"  {d}")
            else:
                if self.validate_defaults:
                    raise ConfigCompositionException(
                        f"No defaults found for {key}"
                    )
                else:
                    self.log.warning(f"No defaults found for {key}")

        if job_type_missing:
            raise ConfigCompositionException(
                "Please provide `job_type=type`. Available types are: train, generate, evaluate"
            )
