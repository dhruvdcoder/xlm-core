from typing import Any
from lightning import Trainer
from lightning.pytorch.callbacks import (
    OnExceptionCheckpoint as _OnExceptionCheckpoint,
)
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class OnExceptionCheckpoint(_OnExceptionCheckpoint):
    """Same as the base class, but skips saving when exceptions are raised during sanity checking."""

    def on_exception(self, trainer: Trainer, *_: Any, **__: Any) -> None:
        if trainer.sanity_checking:
            return
        # don't save checkpoint if the training has not started yet
        if trainer.global_step <= 1:
            return
        logger.info(
            f"Saving checkpoint on exception at {self.ckpt_path} for epoch {trainer.current_epoch} and global step {trainer.global_step}"
        )
        return super().on_exception(trainer, *_, **__)
