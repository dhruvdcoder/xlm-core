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
    Note: We rely on having same number right pads in target and pred, which may not be true for ZLM.
    """

    output_start_idx = loss_dict["output_start_idx"]
    pred = loss_dict["ids"][:, output_start_idx:]
    return {
        "pred": pred,
        "target": batch["target_ids"],
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
    target = batch["target_ids"]
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
        loss_dict: Loss dictionary containing loss (since we don't use batch_loss for ZLM).

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

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary containing nlls.

    Returns:
        Dictionary with perplexity value.
    """
    # Perplexity is exp(mean(nlls))
    nlls = loss_dict["nlls"]
    perplexity = torch.exp(nlls.mean())
    return {
        "value": perplexity,
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
