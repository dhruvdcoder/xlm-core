"""PAPL-style unconditional MLM: fixed lengths (default 100..800 step 100), one length per batch."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Sequence

import torch
from torch.utils.data import IterableDataset

from xlm.datamodule import BaseCollatorInput, Collator, Tokenizer

from .types_mlm import MLMBatch

PAPL_DEFAULT_LENGTHS: tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800)


class PaplUnconditionalMLMDataset(IterableDataset):
    """All-mask sequences; yields ``examples_per_node`` samples per length, in length order."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        examples_per_node: int,
        lengths: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> None:
        # Hydra merges dataset_kwargs with uniref50_packed_mlm (e.g. max_length for MLMEmptyDataset).
        self.tokenizer = tokenizer
        self.examples_per_node = examples_per_node
        self.lengths = tuple(lengths) if lengths is not None else PAPL_DEFAULT_LENGTHS

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        mask_id = self.tokenizer.mask_token_id
        for L in self.lengths:
            for _ in range(self.examples_per_node):
                yield {"input_ids": [mask_id] * L}


class PaplUnconditionalCollator(Collator):
    """Stacks equal-length PAPL examples (no BOS/EOS, no random MLM noise)."""

    def __call__(self, examples: List[BaseCollatorInput]) -> MLMBatch:
        lens = [len(e["input_ids"]) for e in examples]
        L = lens[0]
        if not all(l == L for l in lens):
            raise ValueError(
                f"PaplUnconditionalCollator expects fixed length within a batch; got {lens}"
            )
        input_ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            # Same shape as other MLM collators; unconditional has no ground-truth targets.
            "target_ids": input_ids.clone(),
        }
