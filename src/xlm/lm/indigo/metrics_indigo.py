from typing import Dict, Any
import torch
from .types_indigo import IndigoPredictionDict


def mean_metric_update_fn(
    metric: torch.Tensor,
    new_value: torch.Tensor,
    num_updates: int,
) -> torch.Tensor:
    """Update function for mean metrics."""
    # TODO (URV): Implement mean metric update function
    pass


def custom_indigo_metric(
    predictions: IndigoPredictionDict,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Custom metrics for the Indigo model."""
    # TODO (URV): Implement custom metrics
    return {"perplexity": 0.0, "accuracy": 0.0}
