"""Batch and prediction types for gemma-usdm."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from torch import Tensor as TT


class GemmaUsdmBatch(TypedDict, total=False):
    """Clean seq2seq batch. Noise is applied in the loss, not the collator."""

    input_ids: TT
    attention_mask: TT
    target_ids: TT
    canvas_mask: TT
    fixed_positions_mask: TT


class GemmaUsdmLossDict(TypedDict):
    loss: TT


class GemmaUsdmPredictionDict(TypedDict, total=False):
    text: List[str]
    generated_text: List[str]
    ids: TT
    output_start_idx: int
    steps_taken: TT
    nfe: List[int]
    time_taken: List[float]
    loss: Optional[Any]


class StoredRow(TypedDict):
    input_ids: TT
    attention_mask: TT
    target_ids: TT
    canvas_mask: TT
    soft: TT


Batch = Dict[str, Any]
