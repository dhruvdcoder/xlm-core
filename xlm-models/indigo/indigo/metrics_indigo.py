from typing import Dict, Any
import torch
from .types_indigo import IndigoPredictionDict


def mean_loss_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    return {
        "value": loss_dict["loss"].detach(),
    }
