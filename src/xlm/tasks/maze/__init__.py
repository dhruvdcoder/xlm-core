"""Data preprocessing for the synthetic maze-planning task."""

from typing import Any, Dict, List, Optional

from xlm.datamodule import SimpleSpaceTokenizer

__all__ = ["on_the_fly_fn", "preprocess_fn"]


def preprocess_fn(
    example: Dict[str, Any],
    tokenizer: SimpleSpaceTokenizer,
) -> Dict[str, Any]:
    """Tokenize a maze path and its ordered conditioning subgoals."""
    path: List[int] = example["path"]
    subgoals: List[int] = example["subgoals"]
    subgoal_pos: List[int] = example["subgoal_pos"]
    _validate_anchors(path, subgoals, subgoal_pos)

    example["input_token_ids"] = [
        tokenizer._convert_token_to_id(str(cell)) for cell in path
    ]
    example["prompt_token_ids"] = [
        tokenizer._convert_token_to_id(str(cell)) for cell in subgoals
    ]
    example["anchor_pos"] = list(subgoal_pos)
    return example


def on_the_fly_fn(
    example: Dict[str, Any],
    tokenizer: SimpleSpaceTokenizer,
    block_size: Optional[int] = None,
) -> Dict[str, List[int]]:
    """Expose model-facing path, prompt, and anchor-position fields."""
    del tokenizer
    input_ids: List[int] = example["input_token_ids"]
    prompt_ids: List[int] = example["prompt_token_ids"]
    anchor_pos: List[int] = example["anchor_pos"]

    if block_size is not None and len(input_ids) > block_size:
        raise ValueError(
            f"maze path has {len(input_ids)} tokens, exceeding block_size="
            f"{block_size}; paths must be filtered during generation"
        )
    if len(prompt_ids) != len(anchor_pos):
        raise ValueError(
            "prompt_token_ids and anchor_pos must have equal lengths"
        )
    return {
        "input_ids": list(input_ids),
        "prompt_ids": list(prompt_ids),
        "anchor_pos": list(anchor_pos),
    }


def _validate_anchors(
    path: List[int],
    subgoals: List[int],
    subgoal_pos: List[int],
) -> None:
    if not path:
        raise ValueError("path must be non-empty")
    if len(subgoals) != len(subgoal_pos):
        raise ValueError("subgoals and subgoal_pos must have equal lengths")
    if not subgoals:
        raise ValueError("at least one subgoal is required")
    if any(
        position < 0 or position >= len(path) for position in subgoal_pos
    ):
        raise ValueError("subgoal_pos contains an out-of-range path index")
    if any(
        left >= right
        for left, right in zip(subgoal_pos, subgoal_pos[1:])
    ):
        raise ValueError("subgoal_pos must be strictly increasing")
    anchored_cells = [path[position] for position in subgoal_pos]
    if anchored_cells != subgoals:
        raise ValueError("subgoal_pos does not point to the listed subgoals")
