from typing import Any, Dict
import torch


def postprocess_preds_and_targets_to_same_shape(
    batch: Dict[str, Any],
    loss_dict: Dict[str, Any],
    tokenizer: Any
) -> Dict[str, Any]:

    truncated_preds = []
    truncated_targets = []

    for _pred, _target in zip(loss_dict["ids"], batch["target_ids"]):

        output_start_idx = _pred.tolist().index(tokenizer.bos_token_id) + 1
        pred = _pred[output_start_idx:]
        pred = pred[pred != tokenizer.pad_token_id]
        truncated_preds.append(pred.tolist())

        target = _target[_target != tokenizer.pad_token_id]
        truncated_targets.append(target.tolist())
    
    max_len = max(len(p) for p in truncated_preds + truncated_targets)

    padded_preds = []
    padded_targets = []
    
    for _pred, _target in zip(truncated_preds, truncated_targets):
        
        padded_preds.append(_pred + [tokenizer.pad_token_id] * (max_len - len(_pred)))
        padded_targets.append(_target + [tokenizer.pad_token_id] * (max_len - len(_target)))

    preds = torch.tensor(padded_preds)
    targets = torch.tensor(padded_targets)
    return {
        "pred": preds,
        "target": targets,
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
    """

    return postprocess_preds_and_targets_to_same_shape(batch, loss_dict, tokenizer)


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

    return postprocess_preds_and_targets_to_same_shape(batch, loss_dict, tokenizer)


def mean_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
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
