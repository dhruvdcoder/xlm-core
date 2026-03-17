"""Preprocessing for brozonoyer/gram-graph-coloring.

Dataset has "input" (flattened upper triangle of adjacency matrix) and "target"
(valid 3-coloring per node).
Vocabulary: 0=pad, 1=no edge, 2=edge; 3=red, 4=blue, 5=green.
We produce input_token_ids = [input | target] and prompt_token_ids = [input | mask*len(target)].
"""

from typing import Any, Dict, List

from xlm.datamodule import SimpleSpaceTokenizer
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


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

    # Full sequence: graph structure + coloring
    input_ids: List[int] = list(input_graph) + list(target_coloring)

    # Prompt: graph is given, coloring positions are masked
    prompt_ids: List[int] = list(input_graph) + [mask_id] * len(target_coloring)

    example["input_token_ids"] = input_ids
    example["prompt_token_ids"] = prompt_ids
    return example
