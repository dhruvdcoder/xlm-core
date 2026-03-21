"""UniRef50 (Hugging Face ``airkingbd/uniref50``) preprocessing for protein LM training.

Each row provides a single-letter amino-acid string in column ``seq`` and a
precomputed ``length``. Tokenization should use an ESM-compatible tokenizer
(e.g. ``facebook/esm2_t30_150M_UR50D``) configured in the experiment's
``global_components.tokenizer``.

Very long chains are common; pass ``max_seq_len`` (typically ``block_size``)
via ``preprocess_function_kwargs`` in the dataset config to truncate after
encoding so cached shards stay bounded.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import PreTrainedTokenizerBase


def preprocess_fn(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    seq = example.get("seq")
    if not seq or not isinstance(seq, str):
        example["token_ids"] = []
        return example

    token_ids = tokenizer.encode(seq, add_special_tokens=False)  # type: ignore[arg-type]
    if max_seq_len is not None and len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]

    example["token_ids"] = token_ids
    return example
