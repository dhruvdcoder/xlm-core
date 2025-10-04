import json
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

import hydra
import lightning as L
from lightning.pytorch.loggers import Logger
from omegaconf import ListConfig

from xlm.utils.rank_zero import rank_zero_only, RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class _PredictionWriter:
    """Base class for prediction writers that handle the actual output."""

    supports_reading: bool = False

    def __init__(
        self,
        fields_to_keep_in_output: Optional[List[str]] = None,
    ) -> None:
        """Initialize _PredictionWriter.

        Args:
            fields_to_keep_in_output: List of fields to keep in the output. If None, all fields are kept.
        """
        self.fields_to_keep_in_output = fields_to_keep_in_output

    def filter_fields(self, out_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Filter fields from the output dictionary."""
        if self.fields_to_keep_in_output is None:
            return out_dict

        def keep(k):
            # keep any metrics if present
            if (
                "length" in k
                or "entropy" in k
                or "nll" in k
                or "perplexity" in k
                or "steps" in k
                or "time" in k
            ):
                return True
            for field in self.fields_to_keep_in_output:
                if k.startswith(field) and k != "text_with_spl_tokens":
                    return True
            return False

        return {k: v for k, v in out_dict.items() if keep(k)}

    def write(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth_text: Optional[List[str]],
        step: int,
        epoch: int,
        split: Literal["train", "val", "test", "predict"],
        dataloader_name: str,
        pl_module: L.LightningModule,
        trainer: Optional[L.Trainer],
    ) -> None:
        """Write predictions to the output destination.

        Args:
            predictions: List of prediction dictionaries.
            ground_truth_text: List of ground truth text strings.
            step: Current training step.
            epoch: Current epoch.
            split: The split name (train/val/test/predict).
            dataloader_name: The dataloader name.
            pl_module: The Lightning module.
            trainer: The Lightning trainer.
        """
        raise NotImplementedError

    def attach_loggers(self, loggers: List[Logger]):
        """Attach loggers to the writer if it uses them."""
        pass

    def read(self) -> List[Dict[str, Any]]:
        """Read predictions from a file."""
        if not self.supports_reading:
            raise ValueError(
                "This writer does not support reading predictions"
            )


class FilePredictionWriter(_PredictionWriter):
    """Writer that outputs predictions to JSONL files."""

    def __init__(
        self,
        fields_to_keep_in_output: Optional[List[str]] = None,
        file_path_: Union[
            Path, Literal["none", "from_pl_module"]
        ] = "from_pl_module",
    ) -> None:
        """Initialize FilePredictionWriter.

        Args:
            fields_to_keep_in_output: List of fields to keep in the output. If None, all fields are kept.
            file_path_: Path to the file or special values.
                if "from_pl_module", query the pl_module for the predictions_file for the step and epoch
                set to "none" to disable file writing
        """
        super().__init__(fields_to_keep_in_output)
        self.file_path_ = file_path_
        self.supports_reading = True

    def _get_file_path(
        self,
        pl_module: L.LightningModule,
        split: Literal["train", "val", "test", "predict"],
        dataloader_name: str,
        epoch: int,
        step: int,
    ) -> Optional[Path]:
        file_path: Path
        if self.file_path_ == "from_pl_module":
            file_path = Path(
                pl_module.predictions_dir  # type: ignore
                / f"{split}/{dataloader_name}/{epoch=}_{step=}.jsonl"
            )
        elif self.file_path_ == "none":
            return None
        elif isinstance(self.file_path_, Path):
            file_path = Path(self.file_path_)
        else:
            raise ValueError(f"Invalid file_path_: {self.file_path_}")

        return file_path

    @rank_zero_only
    def write(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth_text: Optional[List[str]],
        step: int,
        epoch: int,
        split: Literal["train", "val", "test", "predict"],
        dataloader_name: str,
        pl_module: L.LightningModule,
        trainer: Optional[L.Trainer],
    ) -> None:
        """Write predictions to a JSONL file.

        Args:
            predictions: List of prediction dictionaries.
            ground_truth_text: List of ground truth text strings.
            step: Current training step.
            epoch: Current epoch.
            split: The split name (train/val/test/predict).
            dataloader_name: The dataloader name.
            pl_module: The Lightning module.
            trainer: The Lightning trainer.
        """
        file_path = self._get_file_path(
            pl_module, split, dataloader_name, epoch, step
        )

        if file_path is None:
            return

        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(file_path, "a") as f:
            for i, dict_ in enumerate(predictions):
                if ground_truth_text:
                    dict_["truth"] = ground_truth_text[i]
                f.write(json.dumps(self.filter_fields(dict_)) + "\n")

    def read(
        self,
        step: int,
        epoch: int,
        split: Literal["train", "val", "test", "predict"],
        dataloader_name: str,
        pl_module: L.LightningModule,
    ) -> List[Dict[str, Any]]:
        """Read predictions from a JSONL file."""
        file_path = self._get_file_path(
            pl_module, split, dataloader_name, epoch, step
        )
        if file_path is None:
            return []
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist")
            return []
        logger.debug(f"Reading predictions from {file_path}")
        with open(file_path, "r") as f:
            return [json.loads(line) for line in f]


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
        super().__init__(fields_to_keep_in_output)
        self.n_rows = n_rows
        self.logger_ = logger_
        self.supports_reading = False

    @rank_zero_only
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
        """Write predictions to Lightning loggers.

        Args:
            predictions: List of prediction dictionaries.
            ground_truth_text: List of ground truth text strings.
            step: Current training step.
            epoch: Current epoch.
            split: The split name (train/val/test/predict).
            dataloader_name: The dataloader name.
            pl_module: The Lightning module.
            trainer: The Lightning trainer.
        """
        if self.logger_ is None:
            logger_ = [
                l_
                for l_ in pl_module.trainer.loggers
                if hasattr(l_, "log_text")
            ]
        else:
            logger_ = self.logger_

        if not logger_:
            return

        # Add truth to predictions first, then filter
        if ground_truth_text is not None:
            predictions_with_truth = []
            for pred, gt in zip(predictions, ground_truth_text):
                pred_with_truth = pred.copy()
                pred_with_truth["truth"] = gt
                predictions_with_truth.append(pred_with_truth)
        else:
            predictions_with_truth = predictions

        # Filter predictions if needed
        filtered_predictions = []
        for pred in predictions_with_truth:
            filtered_pred = self.filter_fields(pred)
            filtered_predictions.append(filtered_pred)

        # Get column names from the first prediction (excluding "truth")
        if filtered_predictions:
            column_names = [
                k for k in filtered_predictions[0].keys() if k != "truth"
            ]
        else:
            column_names = ["text"]  # Default fallback

        # Only log one set of predictions per eval run
        if (
            pl_module.trainer.global_step
            > pl_module.last_global_step_logged_at_which_logged_predictions
        ):
            for logger in logger_:
                # Prepare rows with actual data
                rows = []
                for i, pred in enumerate(filtered_predictions[: self.n_rows]):
                    row = []
                    for col in column_names:
                        row.append(pred.get(col, ""))
                    row.append(pred.get("truth", ""))  # Add ground truth
                    rows.append(row)

                # Add "truth" to column names
                full_column_names = column_names + ["truth"]

                logger.log_text(
                    f"{split}/{dataloader_name}",
                    full_column_names,  # column names
                    rows,  # rows
                    step=step,
                )
            pl_module.last_global_step_logged_at_which_logged_predictions = (
                step
            )


class ConsolePredictionWriter(_PredictionWriter):
    """Writer that outputs predictions to console."""

    def __init__(
        self,
        fields_to_keep_in_output: Optional[List[str]] = None,
    ) -> None:
        """Initialize ConsolePredictionWriter.

        Args:
            fields_to_keep_in_output: List of fields to keep in the output. If None, all fields are kept.
        """
        super().__init__(fields_to_keep_in_output)
        self.supports_reading = False

    @rank_zero_only
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
        """Write predictions to console.

        Args:
            predictions: List of prediction dictionaries.
            ground_truth_text: List of ground truth text strings.
            step: Current training step.
            epoch: Current epoch.
            split: The split name (train/val/test/predict).
            dataloader_name: The dataloader name.
            pl_module: The Lightning module.
            trainer: The Lightning trainer.
        """
        # Print predictions to console
        for i, (dict_, gt) in enumerate(zip(predictions, ground_truth_text)):
            dict_["truth"] = gt
            filtered_dict = self.filter_fields(dict_)

            print("------------")
            for field, value in filtered_dict.items():
                print(f"{field}: {value}")
            print("------------")


class LogPredictions:
    """Main logging class that handles the shared pipeline and delegates to writers."""

    def __init__(
        self,
        writers: Optional[
            Union[
                List[_PredictionWriter],
                List[Literal["file", "logger", "console"]],
            ]
        ] = None,
        inject_target: Optional[str] = None,
        additional_fields_from_batch: Optional[List[str]] = None,
        fields_to_keep_in_output: Optional[List[str]] = None,
    ) -> None:
        """Initialize LogPredictions.

        Args:
            writers: List of prediction writers. If None, creates default writers for backward compatibility.
            inject_target: Key in batch to use as ground truth. If None, empty strings are used.
        """
        self.writers: List[_PredictionWriter] = []
        if isinstance(writers, (list, ListConfig)):
            if isinstance(writers[0], str):
                for writer_type in writers:
                    if writer_type == "file":
                        self.writers.append(
                            FilePredictionWriter(
                                fields_to_keep_in_output=fields_to_keep_in_output
                            )
                        )
                    elif writer_type == "logger":
                        self.writers.append(
                            LoggerPredictionWriter(
                                fields_to_keep_in_output=fields_to_keep_in_output
                            )
                        )
                    elif writer_type == "console":
                        self.writers.append(
                            ConsolePredictionWriter(
                                fields_to_keep_in_output=fields_to_keep_in_output
                            )
                        )
                    else:
                        raise ValueError(f"Invalid writer type: {writer_type}")
            else:
                for writer_config in writers:
                    self.writers.append(hydra.utils.instantiate(writer_config))
        elif writers is None:
            self.writers = []
            logger.warning(
                "No writers provided, no predictions will be logged"
            )
        else:
            raise ValueError(
                "writers must be a list of _PredictionWriter instances or string literals"
                f"Got: {[type(w) for w in writers]}"
            )
        self.inject_target = inject_target
        self.additional_fields_from_batch = additional_fields_from_batch

    def _get_trainer_info(
        self, pl_module: L.LightningModule, trainer: Optional[L.Trainer]
    ) -> tuple[int, int]:
        """Get step and epoch from trainer.

        Args:
            pl_module: The Lightning module.
            trainer: The Lightning trainer.

        Returns:
            Tuple of (step, epoch).
        """
        step = trainer.global_step if trainer is not None else 0
        epoch = trainer.current_epoch if trainer is not None else 0
        return step, epoch

    def _place_additional_fields_in_predictions(
        self, predictions: List[Dict[str, Any]], batch: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Place additional fields in the predictions from the batch.
        Args:
            predictions: The predictions.
            batch: The input batch.

        Returns:
            The predictions with additional fields.
        """
        if self.additional_fields_from_batch is not None:
            for field in self.additional_fields_from_batch:
                for i, pred in enumerate(predictions):
                    pred[field] = batch[field][i]
        return predictions

    def _get_ground_truth_text(
        self, pl_module: L.LightningModule, batch: Dict[str, Any]
    ) -> str:
        """Get ground truth text from batch.

        Args:
            pl_module: The Lightning module.
            batch: The input batch.

        Returns:
            List of ground truth text strings.
        """
        if self.inject_target is not None:
            ground_truth_text = pl_module.tokenizer.batch_decode(
                batch[self.inject_target], skip_special_tokens=True
            )
        else:
            bs = batch["input_ids"].shape[0]
            ground_truth_text = [""] * bs
        return ground_truth_text

    @rank_zero_only
    def __call__(
        self,
        pl_module: L.LightningModule,
        trainer: Optional[L.Trainer],
        batch: Dict[str, Any],
        preds: Dict[str, Any],
        split: Literal["train", "val", "test", "predict"],
        dataloader_name: str,
    ):
        """Log predictions using the shared pipeline and delegate to writers.

        Args:
            pl_module: The Lightning module.
            trainer: The Lightning trainer.
            batch: The input batch.
            preds: The predictions.
            split: The split name (train/val/test/predict).
            dataloader_name: The dataloader name.
        """
        # Shared pipeline: do common work once
        step, epoch = self._get_trainer_info(pl_module, trainer)
        ground_truth_text = self._get_ground_truth_text(pl_module, batch)
        predictions = pl_module.predictor.to_dict(batch, preds)
        # place additional fields in the predictions from the batch
        predictions = self._place_additional_fields_in_predictions(
            predictions, batch
        )

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

    def read(
        self,
        step: int,
        epoch: int,
        split: Literal["train", "val", "test", "predict"],
        dataloader_name: str,
        pl_module: L.LightningModule,
    ) -> List[Dict[str, Any]]:
        """Read predictions from the writers."""
        for writer in self.writers:
            if writer.supports_reading:
                return writer.read(
                    step, epoch, split, dataloader_name, pl_module
                )
        logger.warning(
            "No writer supports reading, returning empty list. "
            "You can add a FilePredictionWriter to your writers list to enable reading."
        )
        return []

    def update_predictions(
        self,
        predictions: List[Dict[str, Any]],
        step: int,
        epoch: int,
        split: Literal["train", "val", "test", "predict"],
        dataloader_name: str,
        pl_module: L.LightningModule,
    ) -> None:
        for writer in self.writers:
            writer.write(
                predictions=predictions,
                ground_truth_text=None,
                step=step,
                epoch=epoch,
                split=split,
                dataloader_name=dataloader_name,
                pl_module=pl_module,
                trainer=None,
            )
