"""Preprocessing for brozonoyer/gram-n-queens.

Dataset has "input" (partial board) and "target" (complete solution).
Vocabulary: 0=pad, 1=empty, 2=queen.
We produce input_token_ids (target) and prompt_token_ids (input with empty→mask).

Token ids use SimpleSpaceTokenizer.for_numbers like sudoku_extreme (via _convert_token_to_id(str(v))).
"""

from typing import Any, Dict, List

from xlm.datamodule import SimpleSpaceTokenizer
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def _dataset_int_to_token_id(v: int, tokenizer: SimpleSpaceTokenizer) -> int:
    """Map dataset vocabulary int to tokenizer id; 0 → pad (not digit \"0\")."""
    if v == 0:
        return int(tokenizer.pad_token_id)
    return tokenizer._convert_token_to_id(str(v))


def gram_n_queens_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: SimpleSpaceTokenizer,
) -> Dict[str, Any]:
    """Preprocess gram-n-queens examples.

    Uses "input" (partial board) and "target" (complete solution).
    Empty cells (dataset 1) in the prompt are replaced with mask_token_id so the model
    knows which positions to predict.
    """
    input_board: List[int] = example["input"]
    target_board: List[int] = example["target"]

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Mask token not found in tokenizer")

    # Dataset: 1 = empty → predict; 2 = queen; 0 = pad
    prompt_ids: List[int] = [
        mask_id if c == 1 else _dataset_int_to_token_id(c, tokenizer)
        for c in input_board
    ]

    input_ids = [_dataset_int_to_token_id(c, tokenizer) for c in target_board]

    example["input_token_ids"] = input_ids
    example["prompt_token_ids"] = prompt_ids
    return example


def gram_n_queens_filter_8x8(example: Dict[str, Any]) -> bool:
    """Keep only 8×8 boards (fixed sequence length for batching)."""
    return example.get("config") == "8x8"


def gram_n_queens_filter_10x10(example: Dict[str, Any]) -> bool:
    """Keep only 10×10 boards (fixed sequence length for batching)."""
    return example.get("config") == "10x10"
