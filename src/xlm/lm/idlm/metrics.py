from typing import Any, Dict
import torch


def mean_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Update function for mean loss metric."""
    return {
        "value": loss_dict["batch_loss"].mean(),
    }


def length_loss_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Update function for length loss metric."""
    return {
        "value": loss_dict["per_example_length_loss"],
    }


def token_ce_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Update function for token cross-entropy metric."""
    return {
        "value": loss_dict["per_example_ce"],
    }


def seq2seq_exact_match_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Update function for sequence-to-sequence exact match metric."""
    # This would be called during prediction, so loss_dict is actually the prediction dict
    preds = loss_dict  # prediction dict
    target_ids = batch.get("target_ids", None)

    if target_ids is None or preds.get("ids", None) is None:
        return {"value": 0.0}

    # Simple exact match computation
    pred_ids = preds["ids"]
    if pred_ids.shape != target_ids.shape:
        return {"value": 0.0}

    exact_matches = (pred_ids == target_ids).all(dim=-1).float()
    return {
        "value": exact_matches,
    }


def seq2seq_token_accuracy_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Update function for sequence-to-sequence token accuracy metric."""
    # This would be called during prediction, so loss_dict is actually the prediction dict
    preds = loss_dict  # prediction dict
    target_ids = batch.get("target_ids", None)

    if target_ids is None or preds.get("ids", None) is None:
        return {"value": 0.0}

    # Token-level accuracy computation
    pred_ids = preds["ids"]

    # Create mask for valid positions (non-padding)
    attention_mask = preds.get("attention_mask", None)
    if attention_mask is not None:
        valid_mask = attention_mask.bool()
    else:
        valid_mask = torch.ones_like(pred_ids, dtype=torch.bool)

    # Compute token accuracy
    correct_tokens = ((pred_ids == target_ids) & valid_mask).sum()
    total_tokens = valid_mask.sum()

    accuracy = correct_tokens.float() / (total_tokens.float() + 1e-8)
    return {
        "value": accuracy,
    }
