# Logging System Refactor

This document describes the refactored logging system that uses composition with a shared pipeline and specialized writers.

## Overview

The original `LogPredictions` class combined both file logging and logger logging in a single class. The new design uses composition where:

- `LogPredictions`: Main class that handles the shared pipeline (common work done once)
- `_PredictionWriter`: Base class for specialized writers that handle actual output
- `FilePredictionWriter`: Handles logging to JSONL files
- `LoggerPredictionWriter`: Handles logging to Lightning loggers
- `ConsolePredictionWriter`: Handles logging to console

## New Architecture

### Composition-Based Design

The key improvement is that common work (getting ground truth text, converting predictions to dict) is done **once** in the shared pipeline, then delegated to specialized writers.

#### LogPredictions (Main Class)
```python
class LogPredictions:
    """Main logging class that handles the shared pipeline and delegates to writers."""
    
    def __init__(
        self,
        writers: Optional[List[_PredictionWriter]] = None,
        inject_target: Optional[str] = None,
    ) -> None:
        """Initialize LogPredictions.
        
        Args:
            writers: List of prediction writers. If None, creates default writers for backward compatibility.
            inject_target: Key in batch to use as ground truth. If None, empty strings are used.
        """
```

#### _PredictionWriter (Base Class)
```python
class _PredictionWriter:
    """Base class for prediction writers that handle the actual output."""
    
    def write(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth_text: List[str],
        step: int,
        epoch: int,
        split: Literal["train", "val", "test", "predict"],
        dataloader_name: str,
        pl_module: L.LightningModule,
        trainer: Optional[L.Trainer],
    ) -> None:
        """Write predictions to the output destination."""
```

### Specialized Writers

#### FilePredictionWriter
```python
class FilePredictionWriter(_PredictionWriter):
    """Writer that outputs predictions to JSONL files."""
    
    def __init__(
        self,
        fields_to_keep_in_output: Optional[List[str]] = None,
        file_path_: Union[Path, Literal["none", "from_pl_module"]] = "from_pl_module",
    ) -> None:
        """Initialize FilePredictionWriter."""
```

#### LoggerPredictionWriter
```python
class LoggerPredictionWriter(_PredictionWriter):
    """Writer that outputs predictions to Lightning loggers."""
    
    def __init__(
        self,
        n_rows: int = 10,
        logger_: Optional[List[Logger]] = None,
        fields_to_keep_in_output: Optional[List[str]] = None,
    ) -> None:
        """Initialize LoggerPredictionWriter.
        
        Args:
            n_rows: Number of rows to log to the logger.
            logger_: List of loggers to use. If None, uses pl_module.trainer.loggers.
            fields_to_keep_in_output: List of fields to keep in the output. If None, all fields are kept.
        """
```

The `LoggerPredictionWriter` dynamically generates column names based on the available fields in the predictions, excluding "truth" which is always added as the last column.

#### ConsolePredictionWriter
```python
class ConsolePredictionWriter(_PredictionWriter):
    """Writer that outputs predictions to console."""
    
    def __init__(
        self,
        fields_to_keep_in_output: Optional[List[str]] = None,
    ) -> None:
        """Initialize ConsolePredictionWriter."""
```

### Backward Compatibility

The original specialized classes are maintained as backward-compatible wrappers:

```python
class FileLogPredictions:
    """Backward-compatible wrapper for file-only logging."""
    
    def __init__(self, fields_to_keep_in_output=None, inject_target=None):
        self.log_predictions = LogPredictions(
            writers=[FilePredictionWriter(fields_to_keep_in_output=fields_to_keep_in_output)],
            inject_target=inject_target,
        )
```

## Usage Examples

### 1. Backward Compatible Usage (Old Way)

```python
# This still works exactly as before
log_predictions = LogPredictions(
    fields_to_keep_in_output=["text", "truth"],
    inject_target="labels"
)
```

### 2. New Composition-Based Usage

#### File-only logging
```python
file_logger = LogPredictions(
    writers=[
        FilePredictionWriter(
            fields_to_keep_in_output=["text", "truth", "confidence"],
        )
    ],
    inject_target="labels"
)
```

#### Logger-only logging
```python
logger_logger = LogPredictions(
    writers=[
        LoggerPredictionWriter(
            n_rows=20,
            fields_to_keep_in_output=["text", "confidence"]
        )
    ],
    inject_target="labels"
)
```

#### Console-only logging
```python
console_logger = LogPredictions(
    writers=[
        ConsolePredictionWriter(
            fields_to_keep_in_output=["text", "truth"]
        )
    ],
    inject_target="labels"
)
```

### 3. Multiple Writers (New Way)

```python
# Create a logger with multiple writers - common work done once!
loggers = LogPredictions(
    writers=[
        FilePredictionWriter(
            fields_to_keep_in_output=["text", "truth"],
        ),
        LoggerPredictionWriter(
            n_rows=10,
            fields_to_keep_in_output=["text", "confidence"]
        ),
        ConsolePredictionWriter(
            fields_to_keep_in_output=["text", "truth"],
        ),
        # You can add more specialized writers here
        # CustomMonitoringWriter(...),
        # DatabaseWriter(...),
    ],
    inject_target="labels"
)
```

### 4. Configuration Examples

#### Single Logger (Backward Compatible)
```yaml
lightning_module:
  log_predictions:
    _target_: xlm.harness.LogPredictions
    fields_to_keep_in_output: ["text", "truth"]
    inject_target: "labels"
```

#### Multiple Writers (New Way)
```yaml
lightning_module:
  log_predictions:
    _target_: xlm.harness.LogPredictions
    writers:
      - _target_: xlm.harness.FilePredictionWriter
        fields_to_keep_in_output: ["text", "truth", "confidence"]
      - _target_: xlm.harness.LoggerPredictionWriter
        n_rows: 15
        fields_to_keep_in_output: ["text", "confidence"]
      - _target_: xlm.harness.ConsolePredictionWriter
        fields_to_keep_in_output: ["text", "truth"]
    inject_target: "labels"
```

#### Using Helper Functions
```yaml
lightning_module:
  log_predictions:
    _target_: xlm.logging_examples.create_combined_logger
    predictions_dir: "${paths.run_dir}/predictions"
    fields_to_keep: ["text", "truth", "confidence"]
    inject_target: "labels"
    n_rows: 15
```

## Implementation Details

### Shared Pipeline

The main `LogPredictions` class handles the shared pipeline:

```python
def __call__(self, pl_module, trainer, batch, preds, split, dataloader_name):
    """Log predictions using the shared pipeline and delegate to writers."""
    # Shared pipeline: do common work once
    step, epoch = self._get_trainer_info(pl_module, trainer)
    ground_truth_text = self._get_ground_truth_text(pl_module, batch)
    predictions = pl_module.predictor.to_dict(batch, preds)

    # Delegate to writers
    for writer in self.writers:
        writer.write(
            predictions=predictions,
            ground_truth_text=ground_truth_text,
            step=step,
            epoch=epoch,
            split=split,
            dataloader_name=dataloader_name,
            pl_module=pl_module,
            trainer=trainer,
        )
```

### Writer Protocol

All writers implement the same interface:

```python
class _PredictionWriter(Protocol):
    def write(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth_text: List[str],
        step: int,
        epoch: int,
        split: Literal["train", "val", "test", "predict"],
        dataloader_name: str,
        pl_module: L.LightningModule,
        trainer: Optional[L.Trainer],
    ) -> None: ...
```

## Benefits

1. **Efficiency**: Common work (ground truth extraction, prediction conversion) is done only once
2. **Separation of Concerns**: Each writer has a single responsibility for output
3. **Flexibility**: Easy to combine different output destinations
4. **Extensibility**: Easy to add new specialized writers
5. **Backward Compatibility**: Existing code continues to work unchanged
6. **Type Safety**: Proper type annotations and protocol definitions

## Performance Improvements

The new design eliminates duplicate work:

**Before (inefficient):**
```python
# Each logger does this work independently
for logger in loggers:
    ground_truth_text = logger._get_ground_truth_text(pl_module, batch)  # Duplicate
    predictions = pl_module.predictor.to_dict(batch, preds)  # Duplicate
    logger.write(...)
```

**After (efficient):**
```python
# Common work done once
ground_truth_text = self._get_ground_truth_text(pl_module, batch)  # Once
predictions = pl_module.predictor.to_dict(batch, preds)  # Once

# Delegated to writers
for writer in self.writers:
    writer.write(predictions, ground_truth_text, ...)
```

## Migration Guide

### For Existing Code

No changes required! The existing `LogPredictions` class maintains the same interface.

### For New Code

1. **Simple cases**: Use the composition-based design with writers
2. **Complex cases**: Create `LogPredictions` with multiple writers
3. **Configuration**: Use the new writer-based configuration format

### Example Migration

**Before (still works):**
```python
log_predictions = LogPredictions(
    fields_to_keep_in_output=["text", "truth"],
    inject_target="labels"
)
```

**After (new way):**
```python
log_predictions = LogPredictions(
    writers=[
        FilePredictionWriter(
            fields_to_keep_in_output=["text", "truth"],
        ),
        LoggerPredictionWriter(
            n_rows=10
        ),
        ConsolePredictionWriter(
            fields_to_keep_in_output=["text", "truth"],
        )
    ],
    inject_target="labels"
)
```

## Future Extensions

The new architecture makes it easy to add new specialized writers:

```python
class DatabasePredictionWriter(_PredictionWriter):
    """Write predictions to a database."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def write(self, predictions, ground_truth_text, step, epoch, split, dataloader_name, pl_module, trainer):
        # Database writing logic here
        pass

class MonitoringPredictionWriter(_PredictionWriter):
    """Send predictions to a monitoring service."""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
    
    def write(self, predictions, ground_truth_text, step, epoch, split, dataloader_name, pl_module, trainer):
        # Monitoring logic here
        pass
```

These can then be easily added to the logging pipeline:

```python
loggers = LogPredictions(
    writers=[
        FilePredictionWriter(...),
        LoggerPredictionWriter(...),
        ConsolePredictionWriter(...),
        DatabasePredictionWriter(connection_string="..."),
        MonitoringPredictionWriter(endpoint="..."),
    ],
    inject_target="labels"
)
``` 