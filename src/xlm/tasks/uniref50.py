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

from typing import Any, Dict, List, Optional

import numpy as np
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


def pack_sequences_fn(
    examples: Dict[str, List[List[int]]],
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    drop_last: bool = True,
    **kwargs: Any,
) -> Dict[str, List[List[int]]]:
    """DPLM-style random crop + EOS packing for UniRef50 protein sequences.

    For each sequence in the batch, randomly crops to ``block_size`` if longer
    (matching the subsampling logic from DPLM's ``UniRefHFDataset.__getitem__``).
    The cropped sequences are then concatenated with EOS separators and chunked
    into blocks of exactly ``block_size`` via ``xlm.datamodule.pack_sequences``.

    Used as ``on_the_fly_group_processor`` in the packed UniRef50 dataset config.
    ``tokenizer`` and ``block_size`` are injected automatically by
    ``DatasetManager``; ``drop_last`` can be overridden via
    ``on_the_fly_group_processor_kwargs``.
    """
    from xlm.datamodule import pack_sequences

    sequences: List[List[int]] = examples["token_ids"]
    cropped: List[List[int]] = []
    for seq in sequences:
        if len(seq) > block_size:
            start = int(np.random.randint(0, len(seq) - block_size + 1))
            seq = seq[start : start + block_size]
        cropped.append(seq)

    return pack_sequences(
        {"token_ids": cropped},
        tokenizer=tokenizer,
        block_size=block_size,
        drop_last=drop_last,
        use_bos=False,
    )
