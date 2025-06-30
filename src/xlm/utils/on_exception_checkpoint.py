from typing import Any
from lightning import Trainer
from lightning.pytorch.callbacks import (
    OnExceptionCheckpoint as _OnExceptionCheckpoint,
)


class OnExceptionCheckpoint(_OnExceptionCheckpoint):
    """Same as the base class, but skips saving when exceptions are raised during sanity checking."""

    def on_exception(self, trainer: Trainer, *_: Any, **__: Any) -> None:
        if trainer.sanity_checking:
            return
        # don't save checkpoint if the training has not started yet
        if trainer.global_step <= 1:
            return
        return super().on_exception(trainer, *_, **__)
