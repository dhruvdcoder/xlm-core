from typing import Any, Dict


def mean_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "value": loss_dict["batch_loss"].mean(),
    }


def length_loss_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "value": loss_dict["per_example_length_loss"],
    }


def token_ce_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "value": loss_dict["per_example_ce"],
    }
