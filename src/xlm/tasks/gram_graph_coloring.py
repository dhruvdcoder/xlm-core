"""Preprocessing for brozonoyer/gram-graph-coloring.

Dataset has "input" (flattened upper triangle of adjacency matrix) and "target"
(valid 3-coloring per node).
Vocabulary: 0=pad, 1=no edge, 2=edge; 3=red, 4=blue, 5=green.
We produce input_token_ids = [input | target] and prompt_token_ids = [input | mask*len(target)].

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


def gram_graph_coloring_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: SimpleSpaceTokenizer,
) -> Dict[str, Any]:
    """Preprocess gram-graph-coloring examples.

    Uses "input" (graph structure, values 1/2) and "target" (coloring, values 3/4/5).
    Full sequence is [input | target]. The graph is given; node colors are to predict.
    """
    input_graph: List[int] = example["input"]
    target_coloring: List[int] = example["target"]

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Mask token not found in tokenizer")

    graph_ids = [_dataset_int_to_token_id(v, tokenizer) for v in input_graph]
    color_ids = [_dataset_int_to_token_id(v, tokenizer) for v in target_coloring]

    input_ids: List[int] = graph_ids + color_ids
    prompt_ids: List[int] = graph_ids + [mask_id] * len(target_coloring)

    example["input_token_ids"] = input_ids
    example["prompt_token_ids"] = prompt_ids
    return example


def gram_graph_coloring_filter_8_vertex(example: Dict[str, Any]) -> bool:
    """Keep only 8-vertex instances (fixed sequence length for batching)."""
    return example.get("config") == "8-vertex"


def gram_graph_coloring_filter_10_vertex(example: Dict[str, Any]) -> bool:
    """Keep only 10-vertex instances (fixed sequence length for batching)."""
    return example.get("config") == "10-vertex"
