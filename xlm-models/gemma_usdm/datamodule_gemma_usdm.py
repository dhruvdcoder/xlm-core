"""Seq2seq collators for gemma-usdm.

The collator emits a clean source plus a fixed-length canvas. Uniform
replacement happens in the loss, on the first visit only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional

import torch
from torch import Tensor as TT

from xlm.datamodule import Collator, Tokenizer
from xlm.utils.nn import pad_truncate_list


def suffix_ids(example: Mapping[str, Any], target_field: str = "target_ids") -> List[int]:
    """Target token ids, falling back to ``input_ids`` from the IWSLT processor."""
    if target_field in example:
        return list(example[target_field])
    if target_field == "target_ids" and "input_ids" in example:
        return list(example["input_ids"])
    return []


def _canvas_token_ids(
    suffix: List[int],
    block_size: int,
    pad_token_id: int,
    eos_token_id: Optional[int],
) -> List[int]:
    """Exactly ``block_size`` canvas ids. EOS is kept when the suffix is truncated."""
    if eos_token_id is None:
        return pad_truncate_list(suffix, block_size, pad_token_id, pad_left=False)
    if len(suffix) + 1 > block_size:
        body = suffix[: block_size - 1] + [eos_token_id]
    else:
        body = suffix + [eos_token_id]
    return pad_truncate_list(body, block_size, pad_token_id, pad_left=False)


def _canvas_targets(
    canvas: List[int],
    loss_on_padding: bool,
    eos_token_id: Optional[int],
    pad_token_id: int,
) -> List[int]:
    """Canvas supervision. EOS is always included. Pads after it are optional."""
    if loss_on_padding:
        return list(canvas)
    targets = list(canvas)
    if eos_token_id is not None and eos_token_id in canvas:
        cut = canvas.index(eos_token_id) + 1
    else:
        cut = len(canvas)
        while cut > 0 and canvas[cut - 1] == pad_token_id:
            cut -= 1
    for j in range(cut, len(canvas)):
        targets[j] = -100
    return targets


class GemmaUsdmTrainCollator(Collator):
    """Right-padded source plus a canvas of exactly ``block_size`` tokens.

    The row is padded to ``input_block_size + block_size`` so every training
    step has the same sequence length and the self-conditioning buffer can
    concatenate the replay half. Attention covers the source and the whole
    canvas, including canvas pads. Slots past that span stay masked.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        input_block_size: int,
        add_bos: bool = False,
        add_eos: bool = True,
        truncate: Literal["max", "block", None] = "block",
        loss_on_padding: bool = True,
        target_field: str = "target_ids",
        prompt_field: str = "prompt_ids",
    ):
        del truncate  # layout is always a fixed source budget plus a fixed canvas
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.input_block_size = input_block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.loss_on_padding = loss_on_padding
        self.target_field = target_field
        self.prompt_field = prompt_field

    def __call__(self, examples: List[Mapping[str, Any]]) -> Dict[str, TT]:
        if self.block_size is None or self.input_block_size is None:
            raise ValueError("block_size and input_block_size are required")
        eos_id = self.tokenizer.eos_token_id if self.add_eos else None
        bos_id = self.tokenizer.bos_token_id if self.add_bos else None
        pad_id = self.tokenizer.pad_token_id
        total = self.input_block_size + self.block_size + int(self.add_bos)

        input_rows: List[List[int]] = []
        target_rows: List[List[int]] = []
        attn_rows: List[List[bool]] = []
        canvas_rows: List[List[bool]] = []
        fixed_rows: List[List[bool]] = []

        for example in examples:
            prefix = list(example[self.prompt_field])
            if len(prefix) > self.input_block_size:
                prefix = prefix[-self.input_block_size :]
            if bos_id is not None:
                prefix = prefix + [bos_id]
            canvas = _canvas_token_ids(
                suffix_ids(example, self.target_field),
                self.block_size,
                pad_id,
                eos_id,
            )
            canvas_targets = _canvas_targets(
                canvas, self.loss_on_padding, eos_id, pad_id
            )
            content_len = len(prefix) + self.block_size
            tail = total - content_len
            input_rows.append(prefix + canvas + [pad_id] * tail)
            target_rows.append(
                [-100] * len(prefix) + canvas_targets + [-100] * tail
            )
            attn_rows.append([True] * content_len + [False] * tail)
            canvas_rows.append(
                [False] * len(prefix) + [True] * self.block_size + [False] * tail
            )
            fixed_rows.append(
                [True] * len(prefix) + [False] * self.block_size + [True] * tail
            )

        attention = torch.tensor(attn_rows, dtype=torch.bool)
        fixed = torch.tensor(fixed_rows, dtype=torch.bool) | ~attention
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": attention,
            "target_ids": torch.tensor(target_rows, dtype=torch.long),
            "canvas_mask": torch.tensor(canvas_rows, dtype=torch.bool),
            "fixed_positions_mask": fixed,
        }


class GemmaUsdmPredCollator(Collator):
    """Left-padded source. The predictor appends the canvas."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        input_block_size: int,
        add_bos: bool = False,
        add_eos: bool = True,
        truncate: Literal["max", "block", None] = "block",
        loss_on_padding: bool = True,
        truncate_long_targets: bool = False,
        target_field: str = "target_ids",
        prompt_field: str = "prompt_ids",
        pass_through_fields: Optional[List[str]] = None,
    ):
        del truncate, loss_on_padding
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.input_block_size = input_block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.truncate_long_targets = truncate_long_targets
        self.target_field = target_field
        self.prompt_field = prompt_field
        self.pass_through_fields = list(
            pass_through_fields
            if pass_through_fields is not None
            else ["target_text", "target_raw", "id"]
        )

    def __call__(self, examples: List[Mapping[str, Any]]) -> Dict[str, Any]:
        if self.input_block_size is None or self.block_size is None:
            raise ValueError("block_size and input_block_size are required")
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id if self.add_eos else None
        prefixes: List[List[int]] = []
        for example in examples:
            prefix = list(example[self.prompt_field])
            if self.add_bos:
                prefix = prefix + [self.tokenizer.bos_token_id]
            prefixes.append(
                pad_truncate_list(
                    prefix,
                    self.input_block_size,
                    pad_id,
                    pad_left=True,
                )
            )
        attention = [
            [token != pad_id for token in prefix] for prefix in prefixes
        ]
        # A prefix that is entirely pad (empty source) still needs a valid
        # position. Left-pad puts the content on the right; an all-pad row
        # has no content, so leave the mask empty.
        suffixes = [suffix_ids(example, self.target_field) for example in examples]
        add_eos = int(self.add_eos)
        raw_max = max((len(s) + add_eos for s in suffixes), default=0)
        if raw_max > self.block_size and not self.truncate_long_targets and any(
            len(s) for s in suffixes
        ):
            raise ValueError(
                f"Max target length {raw_max} exceeds block_size {self.block_size}"
            )
        target_rows = [
            _canvas_token_ids(s, self.block_size, pad_id, eos_id) for s in suffixes
        ]
        batch: Dict[str, Any] = {
            "input_ids": torch.tensor(prefixes, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.bool),
            "target_ids": torch.tensor(target_rows, dtype=torch.long),
        }
        if examples:
            for key in self.pass_through_fields:
                if key in examples[0]:
                    batch[key] = [example[key] for example in examples]
        return batch


def print_batch_gemma_usdm(
    batch: Dict[str, Any],
    split: Literal["train", "val", "test", "predict"],
    tokenizer: Tokenizer,
    dataloader_name: str = "",
) -> None:
    """Print the first row of a gemma-usdm batch."""
    from xlm.utils.rank_zero import RankedLogger

    logger = RankedLogger(__name__, rank_zero_only=True)
    logger.info(
        "Printing first entries of the tensors in batch for %s/%s...",
        split,
        dataloader_name,
    )
    ids = batch["input_ids"][0].tolist()
    print("input tokens:")
    print(tokenizer.decode(ids))
    print("input_ids:")
    print(batch["input_ids"][0])
    if "canvas_mask" in batch:
        print("canvas_mask:")
        print(batch["canvas_mask"][0].int())
    if "target_ids" in batch:
        print("target_ids:")
        print(batch["target_ids"][0])
