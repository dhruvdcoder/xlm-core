import os
import sys
from pathlib import Path
from typing import List, Optional, Set, Dict, Tuple
import logging
import importlib.util

logger = logging.getLogger(__name__)


class ExternalModelConflictError(Exception):
    """Raised when there are conflicts between external models or with core XLM."""

    pass


def validate_model_names(model_names: List[str]) -> None:
    """Validate that model names don't conflict with each other or core XLM.

    Args:
        model_names: List of model names from .xlm_models file

    Raises:
        ExternalModelConflictError: If conflicts are detected
    """
    # Check for duplicates
    seen_names = set()
    duplicates = set()
    for name in model_names:
        if name in seen_names:
            duplicates.add(name)
        seen_names.add(name)

    if duplicates:
        raise ExternalModelConflictError(
            f"Duplicate model names found in .xlm_models: {sorted(duplicates)}"
        )

    # Check for conflicts with core XLM model names
    # NOTE: All models have been migrated to external, so no core conflicts expected
    core_xlm_models = set()  # All models are now external
    conflicts = seen_names.intersection(core_xlm_models)
    if conflicts:
        logger.warning(
            f"External model names conflict with core XLM models: {sorted(conflicts)}. "
            "External models will override core configs."
        )


def validate_python_packages(model_dirs: List[Path]) -> None:
    """Validate that external model Python packages don't conflict.

    Args:
        model_dirs: List of external model directory paths

    Raises:
        ExternalModelConflictError: If conflicts are detected
    """
    package_to_model = {}

    for model_dir in model_dirs:
        model_name = model_dir.name
        pkg_name = model_name
        pkg_path = model_dir

        if pkg_path.exists() and pkg_path.is_dir():
            if pkg_name in package_to_model:
                raise ExternalModelConflictError(
                    f"Python package conflict: Package {pkg_name} already exists"
                )
            package_to_model[pkg_name] = model_name

            # Check if package name conflicts with standard library or common packages
            common_packages = {
                "os",
                "sys",
                "torch",
                "numpy",
                "model",
                "models",
            }
            if pkg_name in common_packages:
                logger.warning(
                    f"External model '{model_name}' uses package name '{pkg_name}' "
                    f"which might conflict with common packages"
                )

def validate_config_structure(model_dirs: List[Path]) -> None:
    """Validate that external model config structures are valid.

    Args:
        model_dirs: List of external model directory paths

    Raises:
        ExternalModelConflictError: If validation fails
    """
    required_configs = ["model", "model_type"]
    optional_configs = ["collator", "experiment", "datamodule"]

    for model_dir in model_dirs:
        model_name = model_dir.name
        config_dir = model_dir / "configs"

        # Check required config groups exist
        missing_required = []
        for config_group in required_configs:
            group_dir = config_dir / config_group
            if not group_dir.exists():
                missing_required.append(config_group)

        if missing_required:
            raise ExternalModelConflictError(
                f"External model '{model_name}' missing required config groups: {missing_required}"
            )

        # Check that required configs have at least one config file
        for config_group in required_configs:
            group_dir = config_dir / config_group
            config_files = list(group_dir.glob("*.yaml"))
            if not config_files:
                raise ExternalModelConflictError(
                    f"External model '{model_name}' has empty required config group: {config_group}"
                )


def validate_external_models(
    model_names: List[str], model_dirs: List[Path], strict: bool = True
) -> None:
    """Comprehensive validation of external models.

    Args:
        model_names: List of model names from .xlm_models file
        model_dirs: List of discovered external model directories
        strict: If True, raise errors for conflicts. If False, just warn.

    Raises:
        ExternalModelConflictError: If validation fails and strict=True
    """
    try:
        validate_model_names(model_names)
        validate_python_packages(model_dirs)
        validate_config_structure(model_dirs)

        logger.debug(
            f"External model validation passed for {len(model_dirs)} models"
        )

    except ExternalModelConflictError as e:
        if strict:
            raise
        else:
            logger.warning(f"External model validation warning: {e}")


def discover_external_models(
    xlm_models_file: Optional[str] = None,
    search_dirs: Optional[List[str]] = None,
    validate: bool = True,
    strict_validation: bool = True,
) -> List[Path]:
    """Discover external model directories by reading .xlm_models file.

    Args:
        xlm_models_file: Path to .xlm_models file. If None, looks for .xlm_models in current directory.
        search_dirs: List of directories to search for model directories. If None, uses current directory.
        validate: Whether to run validation on discovered models.
        strict_validation: If True, raise errors for validation failures. If False, just warn.

    Returns:
        List of paths to external model directories.

    Raises:
        ExternalModelConflictError: If validation fails and strict_validation=True
    """
    if xlm_models_file is None:
        xlm_models_file = ".xlm_models"

    if search_dirs is None:
        search_dirs = set([
            ".",  # Current directory
            "xlm-models",  # Standard xlm-models directory
            os.environ.get("XLM_MODELS_PATH", ""),  # Environment variable
        ])

    model_dirs = []

    # Check if .xlm_models file exists - try multiple locations
    xlm_models_path = None
    possible_locations = [
        Path(xlm_models_file),  # Current directory
        Path("xlm-models") / xlm_models_file,  # In xlm-models directory
    ]

    for location in possible_locations:
        if location.exists():
            xlm_models_path = location
            break

    # Read model names from .xlm_models file if present (during development)
    if xlm_models_path:
        try:
            with open(xlm_models_path, "r") as f:
                model_names = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
        except Exception as e:
            logger.warning(f"Failed to read {xlm_models_file}: {e}")
            return model_dirs
    elif os.environ.get("XLM_MODEL_PACKAGES"):
        model_names = os.environ["XLM_MODEL_PACKAGES"].split(os.pathsep)
        for model_name in model_names:
            package_spec = importlib.util.find_spec(model_name)
            if package_spec:
                pkg_path = Path(package_spec.submodule_search_locations[0])
                search_dirs.add(pkg_path.parent)
    else:   
        logger.info(
            f"No {xlm_models_file} file found in any location, and no environment variable for model packages is set. No external models will be loaded."
        )
        return model_dirs

    if not model_names:
        logger.info(f"No model names found in {xlm_models_file}")
        return model_dirs

    # Search for each declared model in the search directories
    for model_name in model_names:
        found = False
        for search_dir in search_dirs:
            if not search_dir:
                continue
            search_path = Path(search_dir)
            if not search_path.exists():
                continue

            model_dir = search_path / model_name
            if not model_dir.exists() or not model_dir.is_dir():
                continue

            # Validate that this looks like an external model directory
            has_configs = (model_dir / "configs").exists()

            # TODO: Add optional checks to verify Python package files

            # if has_configs:
            if has_configs:
                model_dirs.append(model_dir)
                logger.info(
                    f"Found external model '{model_name}' at: {model_dir}"
                )
                found = True
                break

        if not found:
            logger.warning(
                f"External model '{model_name}' declared in {xlm_models_file} but not found in search paths"
            )

    # Run validation if requested
    if validate and model_dirs:
        validate_external_models(model_names, model_dirs, strict_validation)

    return model_dirs


def setup_external_models(
    validate: bool = True,
    strict_validation: bool = False,  # Default to warnings only
) -> List[Path]:
    """Auto-discover and register external models.

    Args:
        validate: Whether to run validation on discovered models.
        strict_validation: If True, raise errors for validation failures. If False, just warn.

    Returns:
        List of discovered external model directories.

    Raises:
        ExternalModelConflictError: If validation fails and strict_validation=True
    """
    model_dirs = discover_external_models(
        validate=validate, strict_validation=strict_validation
    )
    xlm_model_path = Path("xlm-models")
    if xlm_model_path.exists():
        sys.path.insert(0,str(Path("xlm-models")))
    else:
        for model_dir in model_dirs:
            if model_dir not in sys.path:
                sys.path.insert(0,str(model_dir.parent.resolve()))

    # Log discovered models
    if model_dirs:
        logger.info(
            f"Successfully registered {len(model_dirs)} external models for import"
        )

    return model_dirs
