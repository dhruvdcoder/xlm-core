from typing import Any, Dict
import torch


def seq2seq_exact_match_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Args:
        batch: Dict[str, Any]. Should contain the following keys:
            - "target_ids": Integer[TT, " *batch target_seq_len"]
            - "input_ids": Integer[TT, " *batch input_seq_len"]
        loss_dict: Dict[str, Any]. Should contain the following keys:
            - "ids": Integer[TT, " *batch input_seq_len+target_seq_len"]
    """

    output_start_idx = loss_dict["output_start_idx"]
    pred = loss_dict["ids"][:, output_start_idx:]
    # target = batch["target_ids"][:, output_start_idx:]
    target = batch["target_ids"]
    return {
        "pred": pred,
        "target": target,
        "pred_length": pred.shape[-1],
        "target_length": batch["target_ids"].shape[-1],
    }


def seq2seq_token_accuracy_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Args:
        batch: Dict[str, Any]. Should contain the following keys:
            - "target_ids": Integer[TT, " *batch target_seq_len"]
            - "input_ids": Integer[TT, " *batch input_seq_len"]
        loss_dict: Dict[str, Any]. Should contain the following keys:
            - "ids": Integer[TT, " *batch input_seq_len+target_seq_len"]
    """

    output_start_idx = loss_dict["output_start_idx"]
    pred = loss_dict["ids"][:, output_start_idx:]
    target = batch["target_ids"]
    pred_mask = torch.ones_like(pred, dtype=torch.bool)
    return {
        "pred": pred,
        "target": target,
        "pred_mask": pred_mask,
    }


def mean_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Update function for mean loss metric.

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary containing loss (since we don't use batch_loss for ARLM).

    Returns:
        Dictionary with mean loss value.
    """
    return {
        "value": loss_dict["loss"],
    }
