import os
import re
from datetime import timedelta
from typing import Any, Dict, Literal, Optional, Union
from typing_extensions import override
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from .rank_zero import rank_zero_warn


class ThinningCheckpoint(ModelCheckpoint):
    """Checkpoint callback that saves after every N steps and keeps checkpoints at K*N step intervals.

    Args:
        keep_multiple (int): Keep checkpoints for every keep_multiple * every_n_train_steps
        dirpath: directory to save the model file
        filename: checkpoint filename. Must contain {step} in the pattern
        every_n_train_steps: Number of training steps between checkpoints
        **kwargs: Additional arguments passed to ModelCheckpoint

    Example::
        >>> callback = StepBasedCheckpoint(
        ...     dirpath='checkpoints',
        ...     filename='model-{epoch}-{step}',
        ...     every_n_train_steps=1000,
        ...     keep_multiple=10
        ... )
        # This will:
        # - Save a checkpoint every 1000 steps
        # - Keep checkpoints at steps 10000, 20000, etc.
        # - Delete intermediate checkpoints
    """

    def __init__(
        self,
        dirpath: Optional[Union[Path, str]] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = True,
        save_last: Optional[Union[bool, Literal["link"]]] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
        keep_multiple: int = 1,
    ):
        if filename is not None:
            raise Exception(
                "StepBasedCheckpoint does not support custom filename. You need to use the default {epoch}-{step}.ckpt"
            )
        if every_n_train_steps is None:
            raise Exception(
                "StepBasedCheckpoint requires every_n_train_steps to be set"
            )
        if monitor is not None:
            raise Exception("StepBasedCheckpoint does not support monitor")
        if save_top_k != -1:
            raise Exception("StepBasedCheckpoint does not support save_top_k")

        # Initialize parent class with all arguments
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=None,
            verbose=verbose,
            save_last=save_last,
            save_top_k=-1,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=False,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=None,
            every_n_epochs=None,
            save_on_train_epoch_end=None,
            enable_version_counter=enable_version_counter,
        )

        self.keep_multiple = keep_multiple

        if not self._every_n_train_steps:
            raise Exception(
                "StepBasedCheckpoint requires every_n_train_steps to be set"
            )

        # Prepare the regex pattern once in the constructor
        self._step_extraction_pattern = r"\d+-(\d+)"

    def _extract_step(self, filepath: str) -> Optional[int]:
        """Extract step number from checkpoint filename using the prepared pattern."""
        try:
            match = re.search(
                self._step_extraction_pattern, os.path.basename(filepath)
            )
            if not match:
                return None
            return int(match.group(1))
        except (ValueError, AttributeError, IndexError):
            return None

    def _should_keep_checkpoint(self, filepath: str) -> bool:
        """Check if checkpoint should be kept based on step number."""
        step = self._extract_step(filepath)
        if step is None:
            return True  # Keep checkpoint if we can't parse step number

        keep_interval = self._every_n_train_steps * self.keep_multiple

        # Always keep the most recent checkpoint
        if step == self._last_global_step_saved:
            return True

        return step % keep_interval == 0

    @override
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Save checkpoint and cleanup old ones."""
        super()._save_checkpoint(trainer, filepath)
        self._cleanup_old_checkpoints(trainer)

    def _cleanup_old_checkpoints(self, trainer: "pl.Trainer") -> None:
        """Clean up old checkpoints based on step intervals."""
        # Run cleanup only on rank 0
        # NOTE: Due to this early exit we can't set any state in the rest of this function
        # otherwise the state will diverge across ranks.
        if not trainer.is_global_zero:
            return

        # Get all checkpoints in directory
        if not self.dirpath or not self._fs.exists(self.dirpath):
            return

        checkpoints = [
            f
            for f in self._fs.ls(self.dirpath, detail=False)
            if f.endswith(self.FILE_EXTENSION)
            and self.CHECKPOINT_NAME_LAST
            not in f  # Don't delete 'last' checkpoints
        ]

        # Sort checkpoints by step number to ensure we keep the most recent
        checkpoints_with_step = []
        for ckpt in checkpoints:
            step = self._extract_step(ckpt)
            if step is not None:
                checkpoints_with_step.append((step, ckpt))

        # Sort by step number in descending order
        checkpoints_with_step.sort(reverse=True)

        # Remove checkpoints that don't match our keeping pattern, skipping the most recent
        for step, ckpt in checkpoints_with_step[
            1:
        ]:  # Skip the first (most recent) checkpoint
            if not self._should_keep_checkpoint(ckpt):
                try:
                    self._remove_checkpoint(trainer, ckpt)
                except Exception as e:
                    rank_zero_warn(
                        f"Failed to remove checkpoint {ckpt}: {str(e)}"
                    )

    @override
    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update(
            {
                "keep_multiple": self.keep_multiple,
                "_step_extraction_pattern": self._step_extraction_pattern,
            }
        )
        return state_dict

    @override
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.keep_multiple = state_dict["keep_multiple"]
        self._step_extraction_pattern = state_dict["_step_extraction_pattern"]

    # TODO (checkpointing): Add on_exception hook.
    # See https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/training/src/callbacks/model_checkpoint.py#L14
