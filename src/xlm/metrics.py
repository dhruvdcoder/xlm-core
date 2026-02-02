from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Union,
)
from jaxtyping import Integer, Float
from torchmetrics import Metric, MeanMetric
import torch
from torch import Tensor as TT

from xlm.utils.imports import get_function
from xlm.utils.rank_zero import RankedLogger
import lightning as L

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


class MetricWrapper(torch.nn.Module):
    """Unified metric wrapper that works with both Lightning trainer and Fabric.

    Sends the raw `batch` and `loss_dict` output to the `update_fn` which transforms it into a dict of kwargs for the `metric`. The `update_fn` can contain task specific and model specific logic.

    For Lightning: Use the `log` method to log metrics via LightningModule.
    For Fabric: Use `compute` and `get_log_dict` methods for manual logging.
    """

    def __init__(
        self,
        name: str,
        metric: Metric,
        update_fn: Union[
            Callable[[Dict[str, Any], Dict[str, Any], Any], Dict[str, Any]],
            str,
        ],
        prefix: str = "",
        on_step: bool = False,
        on_epoch: bool = True,
        prog_bar: bool = False,
        add_dataloader_idx: bool = False,
        **pass_to_update_fn,
    ):
        super().__init__()
        self.name = name
        self.metric = metric
        self.prefix = prefix
        self.on_step = on_step
        self.on_epoch = on_epoch
        self.prog_bar = prog_bar
        self.add_dataloader_idx = add_dataloader_idx
        if isinstance(update_fn, str):
            self.update_fn: Callable[
                [Dict[str, Any], Dict[str, Any], Any], Dict[str, Any]
            ] = get_function(update_fn)
        else:
            self.update_fn = update_fn

        if pass_to_update_fn:
            self.pass_to_update_fn = pass_to_update_fn
        else:
            self.pass_to_update_fn = {}
        self._computed_value = None

    def update(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        tokenizer: Any = None,
    ) -> Dict[str, Any]:
        """Update the metric with the current batch and loss_dict."""
        kwargs = self.update_fn(
            batch, loss_dict, tokenizer, **self.pass_to_update_fn
        )
        self.metric.update(**kwargs)
        if hasattr(self.metric, "_computed_value"):
            self._computed_value = self.metric._computed_value
        else:
            self._computed_value = None
        return loss_dict

    @property
    def full_name(self) -> str:
        """Get the full metric name with prefix."""
        return f"{self.prefix}/{self.name}"

    def log(
        self,
        pl_module: L.LightningModule,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
    ):
        """Log the metric using Lightning's logging mechanism."""
        pl_module.log(
            f"{self.prefix}/{self.name}",
            self.metric,
            on_step=self.on_step,
            on_epoch=self.on_epoch,
            prog_bar=self.prog_bar,
            add_dataloader_idx=self.add_dataloader_idx,
        )
        return metrics

    def compute(self) -> torch.Tensor:
        """Compute the current metric value. Useful for Fabric-based training."""
        return self.metric.compute()

    def get_log_dict(self) -> Dict[str, Any]:
        """Get a dictionary with the metric name and computed value for logging. Useful for Fabric-based training."""
        computed_value = self.compute()
        return {
            self.full_name: computed_value,
        }

    def reset(self) -> None:
        """Reset the metric state."""
        self.metric.reset()


def mean_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    return {
        "value": loss_dict["batch_loss"],
    }


def exact_match_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """
    Args:
        batch: Dict[str, Any]. Should contain the following keys:
            - "target_ids": Integer[TT, " *batch target_seq_len"]
        loss_dict: Dict[str, Any]. Should contain the following keys:
            - "ids": Integer[TT, " *batch input_seq_len+target_seq_len"]
    """
    target = batch["target_ids"]
    pred = loss_dict["ids"]
    return {
        "pred": pred,
        "target": target,
        "pred_length": None,
        "target_length": None,
    }


def seq2seq_exact_match_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """
    Args:
        batch: Dict[str, Any]. Should contain the following keys:
            - "target_ids": Integer[TT, " *batch target_seq_len"]
            - "input_ids": Integer[TT, " *batch input_seq_len"]
        loss_dict: Dict[str, Any]. Should contain the following keys:
            - "ids": Integer[TT, " *batch input_seq_len+target_seq_len"]
    Note: We rely on having same number right pads in target and pred, which may not be true for ARLM.
    """
    target = torch.cat([batch["input_ids"], batch["target_ids"]], dim=-1)
    pred = loss_dict["ids"]
    pred_length = pred.shape[-1]
    target = target[:, :pred_length]
    return {
        "pred": pred,
        "target": target,
        "pred_length": None,
        "target_length": None,
    }


def seq2seq_token_accuracy_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """
    Args:
        batch: Dict[str, Any]. Should contain the following keys:
            - "target_ids": Integer[TT, " *batch target_seq_len"]
            - "input_ids": Integer[TT, " *batch input_seq_len"]
        loss_dict: Dict[str, Any]. Should contain the following keys:
            - "ids": Integer[TT, " *batch input_seq_len+target_seq_len"]
    """
    target = torch.cat([batch["input_ids"], batch["target_ids"]], dim=-1)
    pred_mask = torch.cat(
        [
            torch.zeros_like(batch["input_ids"], dtype=torch.bool),
            torch.ones_like(batch["target_ids"], dtype=torch.bool),
        ],
        dim=-1,
    )
    pred = loss_dict["ids"]
    pred_length = pred.shape[-1]
    target = target[:, :pred_length]
    pred_mask = pred_mask[:, :pred_length]
    return {
        "pred": pred,
        "target": target,
        "pred_mask": pred_mask,
    }


################################################################################
# region: metrics


class ExactMatch(MeanMetric):
    def update(
        self,
        pred: Integer[TT, " *batch seq_len"],
        target: Integer[TT, " *batch seq_len"],
        pred_length: Optional[Integer[TT, " *batch"]] = None,
        target_length: Optional[Integer[TT, " *batch"]] = None,
    ):
        """
        Args:
            pred: predicted tokens
            target: target tokens
            pred_length: length of the predicted tokens
            target_length: length of the target tokens
        """
        matches = (
            (pred == target).all(dim=-1).to(torch.float32)
        )  # shape (*batch)
        if pred_length is not None and target_length is not None:
            matches = matches * (pred_length == target_length)
        super().update(matches)
        self._computed_value = matches


class TokenAccuracy(MeanMetric):
    def update(
        self,
        pred: Integer[TT, " *batch seq_len"],
        target: Integer[TT, " *batch seq_len"],
        pred_mask: Optional[Integer[TT, " *batch seq_len"]] = None,
    ):
        """
        Args:
            pred: predicted tokens
            target: target tokens
            pred_mask: True for positions that predicted.
        """
        if pred_mask is None:
            pred_mask = torch.ones_like(pred, dtype=torch.bool)
        temp = (pred == target) * pred_mask
        correct = temp.sum(dim=-1).to(torch.float32)  # shape (*batch)
        total = pred_mask.sum(dim=-1).to(torch.float32)  # shape (*batch)
        acc = correct / total  # shape (*batch)
        super().update(acc)


# endregion: metrics
################################################################################
