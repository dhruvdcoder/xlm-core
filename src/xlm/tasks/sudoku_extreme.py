"""Preprocessing for brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver.

Dataset has "question" (puzzle, "." for blanks) and "answer" (solution).
We convert "." -> "0" to match the tokenizer convention (vocab 0-9) and
produce input_token_ids / prompt_token_ids like the standard sudoku task.
"""

from typing import (
    Any,
    Dict,
    List,
)

from xlm.datamodule import SimpleSpaceTokenizer
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


def _replace(tokens: List[int], zero_id: int, mask_id: int) -> List[int]:
    return [mask_id if c == zero_id else c for c in tokens]


def _normalize_dots(s: str) -> str:
    """Convert '.' (blank) to '0' for tokenizer compatibility."""
    return s.replace(".", "0")


def sudoku_extreme_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: SimpleSpaceTokenizer,
) -> Dict[str, Any]:
    """Preprocess sapientinc-sudoku-extreme examples.

    Uses "question" (puzzle) and "answer" (solution). Blanks are "." in the
    dataset; we convert to "0" before tokenizing.
    
    Also processes "trajectory" field which contains a list of strings
    representing step-by-step board configurations from question to solution.
    """
    partial_sequence: str = _normalize_dots(example["question"])
    ground_truth_sequence: str = _normalize_dots(example["answer"])
    prompt_ids: List[int] = [
        tokenizer._convert_token_to_id(str(ch)) for ch in partial_sequence
    ]
    zero_id = tokenizer._convert_token_to_id("0")
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Mask token not found in tokenizer")
    prompt_ids = _replace(prompt_ids, zero_id, mask_id)
    input_ids = [
        tokenizer._convert_token_to_id(str(ch)) for ch in ground_truth_sequence
    ]
    input_ids = _replace(input_ids, zero_id, mask_id)
    example["input_token_ids"] = input_ids
    example["prompt_token_ids"] = prompt_ids
    
    # Process trajectory if present
    if "trajectory" in example and example["trajectory"] is not None:
        trajectory: List[str] = example["trajectory"]
        trajectory_token_ids: List[List[int]] = []
        for step in trajectory:
            normalized_step: str = _normalize_dots(step)
            step_token_ids: List[int] = [
                tokenizer._convert_token_to_id(str(ch)) for ch in normalized_step
            ]
            step_token_ids = _replace(step_token_ids, zero_id, mask_id)
            trajectory_token_ids.append(step_token_ids)
        example["trajectory_token_ids"] = trajectory_token_ids
    
    return example
