# base on https://github.com/Dao-AILab/flash-attention/blob/main/training/src/callbacks/ema.py
# which is in turn based on the following:
# Inspired by https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/callbacks/stochastic_weight_avg.py
# https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/callbacks/byol_updates.py
# https://forums.pytorchlightning.ai/t/adopting-exponential-moving-average-ema-for-pl-pipeline/488/2
# https://github.com/PyTorchLightning/pytorch-lightning/issues/8100
# https://lightning.ai/forums/t/adopting-exponential-moving-average-ema-for-pl-pipeline/488/3
# https://github.com/Lightning-AI/pytorch-lightning/issues/11688#issuecomment-1027441807

from typing import Dict, Any

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer

from torch_ema import ExponentialMovingAverage


class EMACallback(Callback):
    def __init__(self, decay: float, use_num_updates: bool = True):
        """
        decay: The exponential decay.
        use_num_updates: Whether to use number of updates when computing
            averages.
        """
        super().__init__()
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.ema = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        # It's possible that we already loaded EMA from the checkpoint
        if self.ema is None:
            self.ema = ExponentialMovingAverage(
                [p for p in pl_module.parameters() if p.requires_grad],
                decay=self.decay,
                use_num_updates=self.use_num_updates,
            )
        # make sure the ema is on the same device as the model
        self.ema.to(pl_module.device)

    # Ideally we want on_after_optimizer_step but pytorch-lightning doesn't have it
    # We only want to update when parameters are changing.
    # Because of gradient accumulation, this doesn't happen every training step.
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/11688
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
            self.ema.update()  # pyright: ignore[reportOptionalMemberAccess]

    def on_validation_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # During the initial validation we don't have self.ema yet
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.ema is not None:
            self.ema.restore()

    def on_test_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

    def on_test_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.ema is not None:
            self.ema.restore()

    def on_predict_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

    def on_predict_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.ema is not None:
            self.ema.restore()

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        if self.ema is None:
            return
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        if self.ema is None:
            self.ema = ExponentialMovingAverage(
                [p for p in pl_module.parameters() if p.requires_grad],
                decay=self.decay,
                use_num_updates=self.use_num_updates,
            )
        self.ema.load_state_dict(checkpoint["ema"])
