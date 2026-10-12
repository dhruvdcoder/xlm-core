"""Collators for FlexMDM (Dream-Coder) matching the reference data layout.

Reference layout (``genuine-any-order/FlexMDM``: ``data.py``, ``evals/humaneval_compare/common.py``):

    [BOS] + prompt_ids + [PAD] + answer_ids ... right-padded with PAD to ``max_length``

The PAD right after the prompt is a *kept* separator (part of the prompt mask).
For Dream's tokenizer ``pad == eos == bos == 151643``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch

from xlm.datamodule import Tokenizer
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def encode_flexmdm_prompt_ids(
    prompt_ids: List[int],
    *,
    max_length: int,
    pad_id: int,
    bos_id: Optional[int],
    prepend_bos: bool = True,
) -> List[int]:
    """``[bos] + prompt_ids + [pad]`` prefix; raises if it does not fit (reference behaviour)."""
    prefix: List[int] = []
    if prepend_bos:
        if bos_id is None:
            raise ValueError("prepend_bos=True but tokenizer has no bos_token_id.")
        prefix.append(int(bos_id))
    prefix.extend(int(t) for t in prompt_ids)
    prefix.append(int(pad_id))  # separator
    if len(prefix) >= max_length:
        raise ValueError(
            f"Prompt boundary {len(prefix)} >= max_length {max_length}; shorten the prompt."
        )
    return prefix


def collate_flexmdm_prompt_prefixes(
    prefixes: List[List[int]], *, max_length: int, pad_id: int
) -> Dict[str, torch.Tensor]:
    batch_size = len(prefixes)
    input_ids = torch.full((batch_size, max_length), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for i, prefix in enumerate(prefixes):
        input_ids[i, : len(prefix)] = torch.tensor(prefix, dtype=torch.long)
        attention_mask[i, : len(prefix)] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_mask": attention_mask.clone(),
    }


def encode_flexmdm_prompt_batch(
    tokenizer: Tokenizer,
    prompts: List[str],
    *,
    max_length: int,
    prepend_bos: bool = True,
) -> Dict[str, torch.Tensor]:
    """Tokenize raw prompt strings into a FlexMDM prediction batch (reference ``encode_flexmdm_prompt``)."""
    pad_id = int(tokenizer.pad_token_id)
    prefixes = [
        encode_flexmdm_prompt_ids(
            tokenizer(p, add_special_tokens=False).input_ids,
            max_length=max_length,
            pad_id=pad_id,
            bos_id=tokenizer.bos_token_id,
            prepend_bos=prepend_bos,
        )
        for p in prompts
    ]
    return collate_flexmdm_prompt_prefixes(prefixes, max_length=max_length, pad_id=pad_id)


class FlexMDMDreamPredCollator:
    """Prediction collator: ``[BOS] prompt [PAD]`` prefix, PAD-padded to ``block_size``.

    Expects examples with ``prompt_ids`` (e.g. from
    ``xlm.tasks.humaneval.humaneval_pred_preprocess_fn`` with ``strip=False, sep=""``).
    Emits ``input_ids``, ``attention_mask``, ``prompt_mask`` (and ``fixed`` as an
    alias of ``prompt_mask`` for generic logging code) plus ``pass_through_fields``.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        prepend_bos: bool = True,
        pass_through_fields: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.prepend_bos = prepend_bos
        self.pass_through_fields = (
            list(pass_through_fields)
            if pass_through_fields is not None
            else ["task_id", "sample_k"]
        )

    def __call__(self, examples: List[Mapping[str, Any]]) -> Dict[str, Any]:
        pad_id = int(self.tokenizer.pad_token_id)
        prefixes = [
            encode_flexmdm_prompt_ids(
                list(ex["prompt_ids"]),
                max_length=self.block_size,
                pad_id=pad_id,
                bos_id=self.tokenizer.bos_token_id,
                prepend_bos=self.prepend_bos,
            )
            for ex in examples
        ]
        batch: Dict[str, Any] = collate_flexmdm_prompt_prefixes(
            prefixes, max_length=self.block_size, pad_id=pad_id
        )
        batch["fixed"] = batch["prompt_mask"].long()
        for key in self.pass_through_fields:
            if key in examples[0]:
                batch[key] = [ex[key] for ex in examples]
        return batch


def print_batch_flexmdm_dream(
    batch: Mapping[str, Any],
    split: str,
    tokenizer: Tokenizer,
    dataloader_name: str = "",
) -> None:
    """Debug helper: logs the first example of a FlexMDM (Dream) batch."""
    ids = batch["input_ids"][0]
    attn = batch["attention_mask"][0].bool()
    pmask = batch.get("prompt_mask")
    n_prompt = int(pmask[0].sum()) if pmask is not None else int(attn.sum())
    text = tokenizer.decode(ids[attn], skip_special_tokens=False)
    extra = ""
    if "task_id" in batch:
        extra = f"  task_id={batch['task_id'][0]}  sample_k={batch.get('sample_k', ['?'])[0]}"
    logger.info(
        f"[FlexMDM-Dream] split={split}  dl_name={dataloader_name}  "
        f"shape={tuple(batch['input_ids'].shape)}  attended={int(attn.sum())}  "
        f"prompt_len={n_prompt}{extra}\n{text}"
    )


__all__ = [
    "encode_flexmdm_prompt_ids",
    "print_batch_flexmdm_dream",
    "collate_flexmdm_prompt_prefixes",
    "encode_flexmdm_prompt_batch",
    "FlexMDMDreamPredCollator",
]
