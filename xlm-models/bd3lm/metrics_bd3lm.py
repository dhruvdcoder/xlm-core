"""Metrics computation for Bd3lm model.

This file implements metric update functions used by the training framework.
"""

from typing import Any, Dict, Tuple
import torch


def _align_pred_target(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad pred/target to a common length so they can be compared.

    Generation stops at EOS, so a prediction can be shorter or longer than the
    target. Padding each side with a distinct negative sentinel keeps those
    positions unequal, which makes a length mismatch count as a mismatch
    instead of raising a shape error. Sequences of equal length are untouched.
    """
    len_pred, len_target = pred.shape[-1], target.shape[-1]
    if len_pred == len_target:
        return pred, target
    max_len = max(len_pred, len_target)
    if len_pred < max_len:
        pred = torch.nn.functional.pad(pred, (0, max_len - len_pred), value=-1)
    if len_target < max_len:
        target = torch.nn.functional.pad(
            target, (0, max_len - len_target), value=-2
        )
    return pred, target


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
    Note: We rely on having same number right pads in target and pred, which may not be true for Bd3lm.
    """
    output_start_idx = loss_dict["output_start_idx"]
    pred = loss_dict["ids"][:, output_start_idx:]
    pred, target = _align_pred_target(pred, batch["target_ids"])
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
    output_start_idx = loss_dict["output_start_idx"]
    pred = loss_dict["ids"][:, output_start_idx:]
    pred, target = _align_pred_target(pred, batch["target_ids"])
    pred_mask = torch.ones_like(pred, dtype=torch.bool)
    return {
        "pred": pred,
        "target": target,
        "pred_mask": pred_mask,
    }


def mean_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for mean loss metric.

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary containing loss.

    Returns:
        Dictionary with mean loss value.
    """
    return {
        "value": loss_dict["loss"],
    }


def perplexity_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for perplexity metric.

    Perplexity is exp of the per-token NLL, which is exactly what the loss returns
    as "loss" (nlls.sum() / loss_mask.sum()). For a diffusion model that NLL is a
    variational upper bound, so this is an upper bound on perplexity - the same
    quantity the BD3-LM paper reports as "PPL <=".

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary containing the per-token NLL under "loss".

    Returns:
        Dictionary with perplexity value.
    """
    # Previously read loss_dict["nlls"], which the loss does not return (so this
    # raised KeyError), and averaged it over *all* positions including the unscored
    # ones - dividing by the sequence length instead of by the number of scored
    # tokens. loss_dict["loss"] is already the correctly normalised per-token NLL.
    return {
        "value": torch.exp(loss_dict["loss"]),
    }


def token_nll_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for token-level negative log likelihood metric.

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary containing nlls.

    Returns:
        Dictionary with token-level NLL values.
    """
    return {
        "value": loss_dict["nlls"],
    }


def sequence_length_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for sequence length metric.

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary.

    Returns:
        Dictionary with sequence length values.
    """
    # Calculate sequence lengths based on attention mask
    attention_mask = batch["attention_mask"]
    seq_lengths = attention_mask.sum(dim=1).float()
    return {
        "value": seq_lengths,
    }


def valid_tokens_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for valid tokens count metric.

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary.

    Returns:
        Dictionary with valid tokens count.
    """
    # Count tokens that are not padding (valid tokens)
    attention_mask = batch["attention_mask"]
    target_ids = batch["target_ids"]

    # Valid tokens are those that are not padding and not -100 (ignored tokens)
    valid_tokens = attention_mask & (target_ids != -100)
    valid_token_counts = valid_tokens.sum(dim=1).float()

    return {
        "value": valid_token_counts,
    }
