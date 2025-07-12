"""
Examples of how to use the new composition-based logging system.

This file demonstrates the new logging pipeline design where you can instantiate
LogPredictions with a list of specialized writers that will be called sequentially.
"""

from pathlib import Path
from typing import List

from xlm.harness import (
    LogPredictions,
    FilePredictionWriter,
    LoggerPredictionWriter,
    ConsolePredictionWriter,
)


def create_file_only_logger(
    predictions_dir: Path,
    fields_to_keep: List[str] = None,
    inject_target: str = "labels",
) -> LogPredictions:
    """Create a logger that only writes to JSONL files.

    Args:
        predictions_dir: Directory where prediction files will be saved.
        fields_to_keep: List of fields to keep in the output. If None, all fields are kept.
        inject_target: Key in batch to use as ground truth.

    Returns:
        LogPredictions instance with file writer.
    """
    return LogPredictions(
        writers=[
            FilePredictionWriter(
                fields_to_keep_in_output=fields_to_keep,
            )
        ],
        inject_target=inject_target,
    )


def create_logger_only_logger(
    inject_target: str = "labels",
    n_rows: int = 10,
    fields_to_keep: List[str] = None,
) -> LogPredictions:
    """Create a logger that only logs to Lightning loggers.

    Args:
        inject_target: Key in batch to use as ground truth.
        n_rows: Number of rows to log to the logger.
        fields_to_keep: List of fields to keep in the output. If None, all fields are kept.

    Returns:
        LogPredictions instance with logger writer.
    """
    return LogPredictions(
        writers=[
            LoggerPredictionWriter(
                n_rows=n_rows,
                fields_to_keep_in_output=fields_to_keep,
            )
        ],
        inject_target=inject_target,
    )


def create_combined_logger(
    predictions_dir: Path,
    fields_to_keep: List[str] = None,
    inject_target: str = "labels",
    n_rows: int = 10,
) -> LogPredictions:
    """Create a logger that combines file and logger writers.

    This is the new way to use the logging pipeline. The shared pipeline does
    the common work once (getting ground truth, converting predictions to dict),
    then delegates to each writer for the actual output.

    Args:
        predictions_dir: Directory where prediction files will be saved.
        fields_to_keep: List of fields to keep in the output. If None, all fields are kept.
        inject_target: Key in batch to use as ground truth.
        n_rows: Number of rows to log to the logger.

    Returns:
        LogPredictions instance with multiple writers.
    """
    return LogPredictions(
        writers=[
            FilePredictionWriter(
                fields_to_keep_in_output=fields_to_keep,
            ),
            LoggerPredictionWriter(
                n_rows=n_rows,
                fields_to_keep_in_output=fields_to_keep,
            ),
        ],
        inject_target=inject_target,
    )


def create_custom_logger_chain() -> LogPredictions:
    """Example of a custom logger chain with multiple specialized writers.

    This demonstrates how you can create complex logging pipelines by combining
    multiple specialized writers. The shared pipeline ensures common work is done
    only once.

    Returns:
        LogPredictions instance with custom writer chain.
    """
    return LogPredictions(
        writers=[
            # Log to a specific file with custom fields
            FilePredictionWriter(
                fields_to_keep_in_output=["text", "truth", "confidence"],
            ),
            # Log to logger with more rows for detailed inspection
            LoggerPredictionWriter(
                n_rows=20,
                fields_to_keep_in_output=["text", "confidence"],
            ),
            # Log to console for immediate feedback
            ConsolePredictionWriter(
                fields_to_keep_in_output=["text", "truth"],
            ),
            # You could add more specialized writers here
            # For example, a custom writer that sends data to a monitoring service
            # CustomMonitoringWriter(...),
        ],
        inject_target="labels",
    )


def create_console_only_logger(
    fields_to_keep: List[str] = None,
    inject_target: str = "labels",
) -> LogPredictions:
    """Create a logger that only outputs to console.

    Args:
        fields_to_keep: List of fields to keep in the output. If None, all fields are kept.
        inject_target: Key in batch to use as ground truth.

    Returns:
        LogPredictions instance with console writer.
    """
    return LogPredictions(
        writers=[
            ConsolePredictionWriter(
                fields_to_keep_in_output=fields_to_keep,
            )
        ],
        inject_target=inject_target,
    )


# Example configuration for Hydra
EXAMPLE_CONFIG = {
    "log_predictions": {
        "_target_": "xlm.logging_examples.create_combined_logger",
        "predictions_dir": "${paths.run_dir}/predictions",
        "fields_to_keep": ["text", "truth", "confidence"],
        "inject_target": "labels",
        "n_rows": 15,
    }
}

# Alternative: Use LogPredictions directly with writers
EXAMPLE_CONFIG_DIRECT = {
    "log_predictions": {
        "_target_": "xlm.harness.LogPredictions",
        "writers": [
            {
                "_target_": "xlm.harness.FilePredictionWriter",
                "fields_to_keep_in_output": ["text", "truth"],
            },
            {
                "_target_": "xlm.harness.LoggerPredictionWriter",
                "n_rows": 10,
            },
            {
                "_target_": "xlm.harness.ConsolePredictionWriter",
                "fields_to_keep_in_output": ["text", "truth"],
            },
        ],
        "inject_target": "labels",
    }
}

# Example with custom writer chain
EXAMPLE_CONFIG_CUSTOM = {
    "log_predictions": {
        "_target_": "xlm.logging_examples.create_custom_logger_chain",
    }
}
