# Adapted from https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/gpu_stats_monitor.html#GPUStatsMonitor
# and
# https://github.com/Dao-AILab/flash-attention/blob/main/training/src/callbacks/speed_monitor.py
# We only need the speed monitoring, not the GPU monitoring
import time
from typing import Any, Dict, Literal

from lightning import LightningModule, Trainer, Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric, SumMetric
from torchmetrics.wrappers import Running

from enum import Enum


class TimeEvent(Enum):
    START = True
    END = False


class Timers:
    def __init__(
        self, window: int = 50, reset_value: TimeEvent = TimeEvent.END
    ):
        self.window = window
        self.reset_value = reset_value
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
        self.window_total = 0.0
        self.window_tail = [0.0] * self.window
        self.window_index = 0
        self.current_time = -float("inf")
        self.last_event = self.reset_value.value

    def update(self, event: TimeEvent) -> None:
        if self.last_event == event.value:
            raise ValueError("Time events are not properly paired")

        self.last_event = event.value
        current_time = time.time()

        if event == TimeEvent.START or self.current_time == -float("inf"):
            self.current_time = current_time
        else:
            delta = current_time - self.current_time
            self.total += delta
            self.count += 1

            # Update window total
            self.window_total += delta - self.window_tail[self.window_index]
            self.window_tail[self.window_index] = delta
            self.window_index = (self.window_index + 1) % self.window

            self.current_time = -float("inf")

    def compute(self, prefix: str, postfix: str = "") -> Dict[str, float]:
        if self.count == 0:
            return {
                f"{prefix}/running_mean{postfix}": 0.0,
                f"{prefix}/mean{postfix}": 0.0,
                f"{prefix}/total{postfix}": 0.0,
            }

        mean_time = self.total / self.count
        running_mean_time = self.window_total / min(self.count, self.window)

        return {
            f"{prefix}/running_mean{postfix}": running_mean_time
            * 1000,  # Convert to milliseconds
            f"{prefix}/mean{postfix}": mean_time
            * 1000,  # Convert to milliseconds
            f"{prefix}/total{postfix}": self.total / 60,  # Convert to minutes
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "count": self.count,
            "window_total": self.window_total,
            "window_tail": self.window_tail,
            "window_index": self.window_index,
            "current_time": self.current_time,
            "last_event": self.last_event,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.total = state_dict["total"]
        self.count = state_dict["count"]
        self.window_total = state_dict["window_total"]
        self.window_tail = state_dict["window_tail"]
        self.window_index = state_dict["window_index"]
        self.current_time = state_dict["current_time"]
        self.last_event = state_dict["last_event"]


class SpeedMonitorCallback(Callback):
    """Monitor the speed of each step and each epoch."""

    def __init__(
        self,
        window: int = 50,  # set this to trainer.log_every_n_steps for best results
    ):
        """
        Args:
            window: The window size for the running mean.
        """
        super().__init__()
        # Internal state
        self.intra_step_timer = Timers(
            window=window, reset_value=TimeEvent.END
        )
        # prefix: time/inter_step (ms)
        # inter-step will be first called from on_train_batch_start where "end" will be sent in
        self.inter_step_timer = Timers(
            window=window,
            reset_value=TimeEvent.START,
        )
        self.epoch_timer = Timers(
            window=window,
            reset_value=TimeEvent.END,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "intra_step_timer": self.intra_step_timer.state_dict(),
            "inter_step_timer": self.inter_step_timer.state_dict(),
            "epoch_timer": self.epoch_timer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.intra_step_timer.load_state_dict(state_dict["intra_step_timer"])
        self.inter_step_timer.load_state_dict(state_dict["inter_step_timer"])
        self.epoch_timer.load_state_dict(state_dict["epoch_timer"])

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.epoch_timer.reset()
        self.intra_step_timer.reset()
        self.inter_step_timer.reset()

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.intra_step_timer.reset()
        self.inter_step_timer.reset()
        self.epoch_timer.update(TimeEvent.START)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.epoch_timer.update(TimeEvent.END)
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                self.epoch_timer.compute(prefix="time/epoch", postfix="(min)"),
                step=trainer.global_step,
            )

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # need to reset inter-step timer because there will be a break
        self.inter_step_timer.reset()

    def on_test_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # need to reset inter-step timer because there will be a break
        self.inter_step_timer.reset()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.intra_step_timer.update(TimeEvent.START)
        self.inter_step_timer.update(TimeEvent.END)
        if (
            trainer._logger_connector.should_update_logs
            and trainer.logger is not None
        ):
            trainer.logger.log_metrics(
                self.inter_step_timer.compute(
                    prefix="time/inter_step", postfix="(ms)"
                ),
                step=trainer.global_step,
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.intra_step_timer.update(TimeEvent.END)
        self.inter_step_timer.update(TimeEvent.START)
        if (
            trainer._logger_connector.should_update_logs
            and trainer.logger is not None
        ):
            trainer.logger.log_metrics(
                self.intra_step_timer.compute(
                    prefix="time/intra_step", postfix="(ms)"
                ),
                step=trainer.global_step,
            )
