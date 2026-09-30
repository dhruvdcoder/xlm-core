from typing import Any, Dict

import torch


def exact_match_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    return {
        "pred": loss_dict["ids"],
        "target": batch["target_ids"],
        "pred_length": None,
        "target_length": None,
    }


def infill_token_accuracy_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    pred = loss_dict["ids"]
    target = batch["target_ids"]
    if tokenizer is not None and "input_ids" in batch:
        pred_mask = batch["input_ids"] == tokenizer.mask_token_id
    elif "clamp_mask" in batch:
        pred_mask = ~batch["clamp_mask"].to(dtype=torch.bool)
    else:
        pred_mask = torch.ones_like(pred, dtype=torch.bool)
    return {
        "pred": pred,
        "target": target,
        "pred_mask": pred_mask,
    }


def seq2seq_exact_match_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    output_start_idx = loss_dict["output_start_idx"]
    pred = loss_dict["ids"][:, output_start_idx:]
    target = batch["target_ids"]
    return {
        "pred": pred,
        "target": target,
        "pred_length": pred.shape[-1],
        "target_length": batch["target_ids"].shape[-1],
    }


def seq2seq_token_accuracy_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
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
    return {"value": loss_dict["loss"]}
