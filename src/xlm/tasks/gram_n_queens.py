"""Preprocessing for brozonoyer/gram-n-queens.

Dataset has "input" (partial board) and "target" (complete solution).
Vocabulary: 0=pad, 1=empty, 2=queen.
We produce input_token_ids (target) and prompt_token_ids (input with empty→mask).
"""

from typing import Any, Dict, List

from xlm.datamodule import SimpleSpaceTokenizer
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# Dataset vocab: 0=pad, 1=empty, 2=queen
EMPTY_ID = 1


def gram_n_queens_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: SimpleSpaceTokenizer,
) -> Dict[str, Any]:
    """Preprocess gram-n-queens examples.

    Uses "input" (partial board) and "target" (complete solution).
    Empty cells (1) in the prompt are replaced with mask_token_id so the model
    knows which positions to predict.
    """
    input_board: List[int] = example["input"]
    target_board: List[int] = example["target"]

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Mask token not found in tokenizer")

    # Prompt: partial board, empty cells (1) → mask
    prompt_ids = [
        mask_id if c == EMPTY_ID else c
        for c in input_board
    ]

    # Input/target: full solution (0, 1, 2)
    input_ids = list(target_board)

    example["input_token_ids"] = input_ids
    example["prompt_token_ids"] = prompt_ids
    return example
