from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
)

from torchmetrics import Metric


import lightning as L


class MetricWrapper:
    def __init__(
        self,
        name: str,
        metric: Metric,
        prefix: str = "",
        on_step: bool = False,
        on_epoch: bool = True,
        prog_bar: bool = False,
        add_dataloader_idx: bool = False,
    ):
        self.name = name
        self.metric = metric
        self.prefix = prefix
        self.on_step = on_step
        self.on_epoch = on_epoch
        self.prog_bar = prog_bar
        self.add_dataloader_idx = add_dataloader_idx

    def update(
        self, batch: Dict[str, Any], loss_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def log(
        self,
        pl_module: L.LightningModule,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
    ):
        pl_module.log(
            f"{self.prefix}/{self.metric.name}",
            self.metric,
            on_step=self.on_step,
            on_epoch=self.on_epoch,
            prog_bar=self.prog_bar,
        )
        return metrics

    def reset(self) -> None:
        self.metric.reset()
