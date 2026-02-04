import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Dict, Tuple
import logging
import hydra
from omegaconf import OmegaConf
import omegaconf
import yaml

logger = logging.getLogger(__name__)
# Set to INFO level explicitly since this module runs before Hydra configures logging
logger.setLevel(logging.INFO)
# Ensure there's a handler to output messages (will be reconfigured by Hydra later)
if not logger.handlers and not logging.getLogger().handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

XLM_MODEL_NAMES_FILE = ".xlm_models"
STANDARD_XLM_MODELS_DIRS = ["xlm-models"]
# locations for external models
ENV_XLM_MODELS_PATH = "XLM_MODELS_PATH"  # dir containing external models
ENV_XLM_MODELS_PACKAGES = "XLM_MODELS_PACKAGES"  # installed python packages containing external models, comma separated list of package names
CORE_XLM_MODELS = (
    "arlm:mlm:ilm:mdlm"  # core models available in xlm-models package
)


class ExternalModelConflictError(Exception):
    """Raised when there are conflicts between external models or with core XLM."""

    pass


def discover_external_models() -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Discover external model directories.

    Each model must be packaged with its configs in the following structure:
    <model_root_dir>
       |-- <model_name> # python files
       |   |-- model.py
       |   |-- loss.py
       |   |-- predictor.py
       |   |-- datamodule.py
       |   |-- metrics.py
       |   |-- ..
       |-- configs/
       |   |-- datamodule/
       |   |-- experiment/
       |   |-- ..
       |   |-- ..
       |-- setup.py (optional)
       |-- README.md (optional)


    The model can be installed as a python package (recommended for sharing) or simply kept as a directory (during development).
    When installed as a python package, make sure to package the configs as well like so in the setup.py:
    package_dir={
        "<model_name>": "<model_root_dir>",
    },
    package_data={
        "<model_name>": ["configs/**/*.yaml", "configs/**/*.yml"],
    }

    Discovery:

      1. **search_dirs**: We look for directories that may contain <model_root_dir>.
        By default these are the "." (current directory), "xlm-models" (standard xlm-models directory), and the directory specified in the XLM_MODELS_PATH environment variable.
      2. Each search_dir may contain multiple <model_root_dir> directories, hence multiple models. Therefore, we look for a xlm_models.json file in each search_dir that
        has the following structure:
        ```json
        {
            "<model_name_1>": "<model_root_dir_1>",
            "<model_name_2>": "<model_root_dir_2>",
            ...
        }
        ```
        The model root dir path is relative to the search_dir.
       3. **installed python packages**: We also allow discovering models from installed python packages. The package names must be specified in the XLM_MODELS_PACKAGES environment variable as a colon-separated list (e.g., arlm:mlm:ilm:mdlm). The installed package much follow the same structure as the <model_root_dir> and package the configs as shown above. For the case of python packages, we expect only one model per package.

    Args:
        validate: Whether to run validation on discovered models.
        strict_validation: If True, raise errors for validation failures. If False, just warn.

    Returns:
        dict of model_name -> model_root_dir

    Raises:
        ExternalModelConflictError: If validation fails and strict_validation=True
    """
    final_model_dirs = {}
    model_package_dirs = {}
    search_dirs = [
        ".",  # Current directory
        *STANDARD_XLM_MODELS_DIRS,  # Standard xlm-models directory
        *[
            p for p in os.environ.get(ENV_XLM_MODELS_PATH, "").split(":") if p
        ],  # Environment variable
    ]

    def check_conflict(model_dirs: Dict[str, Path]) -> None:
        for model_name, model_root_dir in model_dirs.items():
            if model_name in final_model_dirs:
                raise ExternalModelConflictError(
                    f"Duplicate model name: {model_name} with root dir {model_root_dir} and {final_model_dirs[model_name]}"
                )
            if model_name in model_package_dirs:
                raise ExternalModelConflictError(
                    f"Duplicate model name: {model_name} with root dir {model_root_dir} and {model_package_dirs[model_name]}"
                )
            final_model_dirs[model_name] = model_root_dir

    def process_search_dir(search_dir: str) -> Dict[str, Path]:
        model_dirs = {}
        xlm_models_file = Path(search_dir) / "xlm_models.json"
        if xlm_models_file.exists():
            try:
                with open(xlm_models_file, "r") as f:
                    xlm_models = json.load(f)
                    for model_name, rel_model_root_dir in xlm_models.items():
                        model_root_dir = Path(search_dir) / rel_model_root_dir
                        model_dirs[model_name] = model_root_dir
            except Exception as e:
                logger.warning(f"Failed to read {xlm_models_file}: {e}")
        return model_dirs

    for search_dir in search_dirs:
        model_dirs = process_search_dir(search_dir)

    check_conflict(model_dirs)
    final_model_dirs.update(model_dirs)

    # process python packages
    package_names = os.environ.get(ENV_XLM_MODELS_PACKAGES, "").split(":")
    package_names = CORE_XLM_MODELS.split(":") + package_names
    if package_names:
        for package_name in package_names:
            try:
                package_spec = importlib.util.find_spec(package_name)
                if package_spec:
                    package_dir = package_spec.submodule_search_locations
                    if package_dir:
                        package_dir = package_dir[0]
                    check_conflict({package_name: Path(package_dir)})
                    model_package_dirs[package_name] = Path(package_dir)
            except Exception as e:
                logger.warning(f"Failed to import {package_name}: {e}")
                continue

    return final_model_dirs, model_package_dirs


def setup_external_models() -> List[Path]:
    """Auto-discover and register external models.

    Args:
        validate: Whether to run validation on discovered models.
        strict_validation: If True, raise errors for validation failures. If False, just warn.

    Returns:
        List of discovered external model directories.

    Raises:
        ExternalModelConflictError: If validation fails and strict_validation=True
    """
    model_dirs, package_dirs = discover_external_models()
    # only model dirs are placed in sys.path for imports, installed packages are imporable anyway
    for model_dir in model_dirs:
        # Add to Python path for imports
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
            logger.info(f"Added to sys.path: {model_dir}")

    # Log discovered models
    logger.info(
        f"Discovered {len(model_dirs)} external models as directories."
    )
    for model_dir, p in model_dirs.items():
        logger.info(f"Model: {model_dir} at {p}")
    logger.info(
        f"Discovered {len(package_dirs)} external models as installed packages."
    )
    for package_dir, p in package_dirs.items():
        logger.info(f"Package: {package_dir}")
    dirs = [p for p in model_dirs.values()] + [
        p for p in package_dirs.values()
    ]
    return dirs


def get_external_commands(
    model_dirs: List[Path],
) -> Dict[str, Callable[omegaconf.DictConfig, Any]]:
    commands_: Dict[str, Callable[omegaconf.DictConfig, Any]] = {}
    for model_dir in model_dirs:
        commands_file = model_dir / "configs" / "commands.yaml"
        if commands_file.exists():
            with open(commands_file, "r") as f:
                commands = yaml.safe_load(f) or {}
                # {"command_name": "<fully qualified function name>"} eg {"ilm_foo": "ilm.commands.foo"}
            for name, function_name in commands.items():
                if name in commands_:
                    raise ExternalModelConflictError(
                        f"Duplicate command name: {name} in {commands_file} and {commands_[name]}"
                    )
                commands_[name] = hydra.utils.get_method(function_name)
                logger.debug(
                    f"Located external command: {name} at {function_name}"
                )
    return commands_
