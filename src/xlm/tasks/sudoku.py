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

def sudoku_preprocess_fn(
    example: Dict[str, Any], tokenizer: SimpleSpaceTokenizer, mask_token: str = "0"
) -> Dict[str, Any]:
    partial_sequence: str = example["quizzes"]
    ground_truth_sequence: str = example["solutions"]
    prompt_ids: List[int] = [tokenizer._convert_token_to_id(str(ch)) for ch in partial_sequence]
    # "0" in partial_sequence represents the blank positions
    zero_id = tokenizer._convert_token_to_id("0")
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Mask token not found in tokenizer")
    prompt_ids = _replace(prompt_ids, zero_id, mask_id)
    input_ids = [tokenizer._convert_token_to_id(str(ch)) for ch in path]
    input_ids = _replace(input_ids, zero_id, mask_id)
    example["input_token_ids"] = input_ids
    example["prompt_token_ids"] = prompt_ids
    return example