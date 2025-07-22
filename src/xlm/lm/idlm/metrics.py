from typing import Any, Dict
import torch
from jaxtyping import Integer
from torch import Tensor as TT


def mean_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for mean loss metric."""
    return {
        "value": loss_dict["batch_loss"].mean(),
    }


def length_loss_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for length loss metric."""
    return {
        "value": loss_dict["per_example_length_loss"],
    }


def token_ce_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for token cross-entropy metric."""
    return {
        "value": loss_dict["per_example_ce"],
    }


def _extend(
    a: Integer[TT, " *batch seq_len"],
    b: Integer[TT, " *batch seq_len"],
    pad_token_id: int,
) -> Integer[TT, " *batch max_seq_len"]:
    """Extend the length of a to the length of b."""
    max_seq_len = max(a.shape[-1], b.shape[-1])
    a = torch.cat(
        [
            a,
            torch.full(
                (a.shape[0], max_seq_len - a.shape[1]),
                pad_token_id,
                dtype=a.dtype,
                device=a.device,
            ),
        ],
        dim=-1,
    )
    b = torch.cat(
        [
            b,
            torch.full(
                (b.shape[0], max_seq_len - b.shape[1]),
                pad_token_id,
                dtype=b.dtype,
                device=b.device,
            ),
        ],
        dim=-1,
    )
    return a, b


def seq2seq_exact_match_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for sequence-to-sequence exact match metric.
    Args:
        batch: Dict[str, Any]. Should contain the following keys:
            - "target_ids": Integer[TT, " *batch target_seq_len"]
            - "input_ids": Integer[TT, " *batch input_seq_len"]
        loss_dict: Dict[str, Any]. Should contain the following keys:
            - "ids": Integer[TT, " *batch input_seq_len+target_seq_len"]
    Note: We rely on having same number right pads in target and pred, which may not be true for ARLM.
    """
    preds = loss_dict  # prediction dict
    target_ids = batch["target_ids"]
    input_end_idx = batch["input_ids"].shape[-1]
    pred_ids = preds["ids"][:, input_end_idx:]
    # extend the length to the longer of the two
    pred_ids, target_ids = _extend(
        pred_ids, target_ids, tokenizer.pad_token_id
    )
    assert pred_ids.shape == target_ids.shape

    return {
        "pred": pred_ids,
        "target": target_ids,
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
    preds = loss_dict  # prediction dict
    target_ids = batch["target_ids"]

    input_end_idx = batch["input_ids"].shape[-1]
    pred_ids = preds["ids"][:, input_end_idx:]
    pred_ids, target_ids = _extend(
        pred_ids, target_ids, tokenizer.pad_token_id
    )
    assert pred_ids.shape == target_ids.shape

    return {
        "pred": pred_ids,
        "target": target_ids,
        "pred_mask": None,
    }
