"""Collators for Flow Reasoning Models."""

from typing import Any, Dict, List, Literal, Optional

import torch
from torch import Tensor as TT

from xlm.datamodule import (
    BaseCollatorInput,
    Collator,
    Seq2SeqCollatorInput,
    Tokenizer,
)
from xlm.noise import NoiseSchedule
from xlm.utils.nn import pad_truncate_list
from xlm.utils.rank_zero import RankedLogger

from .types_frm import FRMBatch

logger = RankedLogger(__name__, rank_zero_only=True)


def _pad_ids(
    seqs: List[List[int]],
    max_seq_len: int,
    pad_id: int,
    pad_left: bool = False,
) -> TT:
    return torch.tensor(
        [
            pad_truncate_list(seq, max_seq_len, pad_id, pad_left=pad_left)
            for seq in seqs
        ],
        dtype=torch.long,
    )


def _resolve_max_len(
    seqs: List[List[int]],
    block_size: Optional[int],
    truncate: Literal["max", "block", None],
) -> int:
    max_in_batch = max((len(s) for s in seqs), default=0)
    if truncate == "block":
        if block_size is None:
            raise ValueError("block_size is required when truncate='block'")
        return block_size
    if truncate == "max":
        if block_size is None:
            return max_in_batch
        return max(max_in_batch, block_size)
    return max_in_batch


def _add_specials(
    seq: List[int],
    bos_id: Optional[int],
    eos_id: Optional[int],
) -> List[int]:
    out = list(seq)
    if bos_id is not None:
        out = [bos_id] + out
    if eos_id is not None:
        out = out + [eos_id]
    return out


def frm_pad_collate(
    input_seqs: List[List[int]],
    target_seqs: List[List[int]],
    clamp_seqs: List[List[bool]],
    clue_seqs: List[List[int]],
    *,
    pad_token_id: int,
    block_size: Optional[int],
    truncate: Literal["max", "block", None],
    loss_on_padding: bool,
    pad_left: bool = False,
) -> FRMBatch:
    max_len = _resolve_max_len(input_seqs, block_size, truncate)
    input_ids = _pad_ids(input_seqs, max_len, pad_token_id, pad_left=pad_left)
    target_ids = _pad_ids(target_seqs, max_len, pad_token_id, pad_left=pad_left)
    clue_ids = _pad_ids(clue_seqs, max_len, pad_token_id, pad_left=pad_left)
    clamp_mask = torch.tensor(
        [
            pad_truncate_list(
                [int(v) for v in c],
                max_len,
                0,
                pad_left=pad_left,
            )
            for c in clamp_seqs
        ],
        dtype=torch.bool,
    )
    if pad_left:
        attention_mask = input_ids != pad_token_id
        # left pads are not clues; keep clamp false there (already 0-padded)
    else:
        attention_mask = torch.tensor(
            [
                pad_truncate_list(
                    [1] * len(seq), max_len, 0, pad_left=False
                )
                for seq in input_seqs
            ],
            dtype=torch.bool,
        )
    if not loss_on_padding:
        target_ids = target_ids.clone()
        target_ids[~attention_mask] = -100
    return FRMBatch(
        input_ids=input_ids,
        attention_mask=attention_mask.to(dtype=torch.long),
        target_ids=target_ids,
        clamp_mask=clamp_mask,
        clue_ids=clue_ids,
    )


class DefaultFRMCollator(Collator):
    """LM-style collator: noise the full ``input_ids`` sequence, no clues."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: Optional[NoiseSchedule] = None,
        loss_on_padding: bool = True,
        truncate: Literal["max", "block", None] = "block",
        add_bos: bool = False,
        add_eos: bool = False,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.loss_on_padding = loss_on_padding
        self.truncate = truncate
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __call__(self, examples: List[BaseCollatorInput]) -> FRMBatch:
        bos = self.tokenizer.bos_token_id if self.add_bos else None
        eos = self.tokenizer.eos_token_id if self.add_eos else None
        seqs = [
            _add_specials(list(e["input_ids"]), bos, eos) for e in examples
        ]
        false_clamp = [[False] * len(s) for s in seqs]
        pad_clues = [[self.tokenizer.pad_token_id] * len(s) for s in seqs]
        return frm_pad_collate(
            seqs,
            seqs,
            false_clamp,
            pad_clues,
            pad_token_id=self.tokenizer.pad_token_id,
            block_size=self.block_size,
            truncate=self.truncate,
            loss_on_padding=self.loss_on_padding,
        )


class DefaultInfillFRMCollator(Collator):
    """Sudoku-style train collator: solution is ``y``, prompt cells are clues."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: Optional[NoiseSchedule] = None,
        loss_on_padding: bool = True,
        truncate: Literal["max", "block", None] = "block",
        add_bos: bool = False,
        add_eos: bool = False,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.loss_on_padding = loss_on_padding
        self.truncate = truncate
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __call__(self, examples: List[BaseCollatorInput]) -> FRMBatch:
        bos = self.tokenizer.bos_token_id if self.add_bos else None
        eos = self.tokenizer.eos_token_id if self.add_eos else None
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id
        solutions = [
            _add_specials(list(e["input_ids"]), bos, eos) for e in examples
        ]
        prompts = [
            _add_specials(list(e["prompt_ids"]), bos, eos) for e in examples
        ]
        clamp_seqs = [[tok != mask_id for tok in p] for p in prompts]
        clue_seqs = [
            [tok if tok != mask_id else pad_id for tok in p] for p in prompts
        ]
        batch = frm_pad_collate(
            solutions,
            solutions,
            clamp_seqs,
            clue_seqs,
            pad_token_id=pad_id,
            block_size=self.block_size,
            truncate=self.truncate,
            loss_on_padding=self.loss_on_padding,
        )
        return batch


class InfillWithTargetPredFRMCollator(Collator):
    """Infill prediction: ``input_ids`` is the prompt; ``target_ids`` is the solution."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: Optional[NoiseSchedule] = None,
        loss_on_padding: bool = False,
        truncate: Literal["max", "block", None] = "block",
        add_bos: bool = False,
        add_eos: bool = False,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.loss_on_padding = loss_on_padding
        self.truncate = truncate
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __call__(self, examples: List[BaseCollatorInput]) -> FRMBatch:
        bos = self.tokenizer.bos_token_id if self.add_bos else None
        eos = self.tokenizer.eos_token_id if self.add_eos else None
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id
        prompts = [
            _add_specials(list(e["prompt_ids"]), bos, eos) for e in examples
        ]
        solutions = [
            _add_specials(list(e["input_ids"]), bos, eos) for e in examples
        ]
        clamp_seqs = [[tok != mask_id for tok in p] for p in prompts]
        clue_seqs = [
            [tok if tok != mask_id else pad_id for tok in p] for p in prompts
        ]
        batch = frm_pad_collate(
            prompts,
            solutions,
            clamp_seqs,
            clue_seqs,
            pad_token_id=pad_id,
            block_size=self.block_size,
            truncate=self.truncate,
            loss_on_padding=self.loss_on_padding,
        )
        return batch


class FRMSeq2SeqTrainCollator(Collator):
    """Concatenate prompt (clamped) and target (noised)."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: Optional[int] = None,
        input_block_size: Optional[int] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        add_bos: bool = False,
        add_eos: bool = True,
        truncate: Literal["max", "block", None] = "block",
        loss_on_padding: bool = True,
        target_field: str = "target_ids",
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.input_block_size = input_block_size
        self.noise_schedule = noise_schedule
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.truncate = truncate
        self.loss_on_padding = loss_on_padding
        self.target_field = target_field

    def _suffix(self, example: Seq2SeqCollatorInput) -> List[int]:
        if self.target_field in example:
            return list(example[self.target_field])  # type: ignore[index]
        return list(example["input_ids"])

    def __call__(self, examples: List[Seq2SeqCollatorInput]) -> FRMBatch:
        bos = self.tokenizer.bos_token_id if self.add_bos else None
        eos = self.tokenizer.eos_token_id if self.add_eos else None
        pad_id = self.tokenizer.pad_token_id
        prefixes = [list(e["prompt_ids"]) for e in examples]
        suffixes = [self._suffix(e) for e in examples]
        if bos is not None:
            prefixes = [[bos] + p for p in prefixes]
        if eos is not None:
            suffixes = [s + [eos] for s in suffixes]
        max_total = None
        if self.truncate == "block":
            prefix_budget = self.input_block_size or 0
            suffix_budget = self.block_size or 0
            max_total = prefix_budget + suffix_budget
        concat = [p + s for p, s in zip(prefixes, suffixes)]
        clamp = [
            [True] * len(p) + [False] * len(s)
            for p, s in zip(prefixes, suffixes)
        ]
        clues = [
            p + [pad_id] * len(s) for p, s in zip(prefixes, suffixes)
        ]
        return frm_pad_collate(
            concat,
            concat,
            clamp,
            clues,
            pad_token_id=pad_id,
            block_size=max_total,
            truncate=self.truncate if max_total is not None else None,
            loss_on_padding=self.loss_on_padding,
        )


class FRMSeq2SeqPredCollator(Collator):
    """Left-padded prompt for seq2seq generation; target is the suffix only."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: Optional[int] = None,
        input_block_size: Optional[int] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        add_bos: bool = False,
        add_eos: bool = True,
        truncate: Literal["max", "block", None] = "block",
        loss_on_padding: bool = True,
        target_field: str = "target_ids",
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.input_block_size = input_block_size
        self.noise_schedule = noise_schedule
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.truncate = truncate
        self.loss_on_padding = loss_on_padding
        self.target_field = target_field

    def _suffix(self, example: Seq2SeqCollatorInput) -> List[int]:
        if self.target_field in example:
            return list(example[self.target_field])  # type: ignore[index]
        if "input_ids" in example:
            return list(example["input_ids"])
        return []

    def __call__(self, examples: List[Seq2SeqCollatorInput]) -> FRMBatch:
        bos = self.tokenizer.bos_token_id if self.add_bos else None
        eos = self.tokenizer.eos_token_id if self.add_eos else None
        pad_id = self.tokenizer.pad_token_id
        prefixes = [list(e["prompt_ids"]) for e in examples]
        if bos is not None:
            prefixes = [[bos] + p for p in prefixes]
        suffixes = [self._suffix(e) for e in examples]
        if eos is not None:
            suffixes = [s + [eos] for s in suffixes]
        prefix_len = self.input_block_size
        if prefix_len is None:
            prefix_len = _resolve_max_len(prefixes, None, None)
        suffix_len = self.block_size or _resolve_max_len(suffixes, None, None)
        input_ids = _pad_ids(prefixes, prefix_len, pad_id, pad_left=True)
        attention_mask = input_ids != pad_id
        clamp_mask = attention_mask.clone()
        clue_ids = input_ids.clone()
        target_ids = _pad_ids(suffixes, suffix_len, pad_id, pad_left=False)
        if not self.loss_on_padding:
            target_attn = target_ids != pad_id
            target_ids = target_ids.clone()
            target_ids[~target_attn] = -100
        return FRMBatch(
            input_ids=input_ids,
            attention_mask=attention_mask.to(dtype=torch.long),
            target_ids=target_ids,
            clamp_mask=clamp_mask,
            clue_ids=clue_ids,
        )


def _replace_100_with_pad(ids: torch.Tensor, tokenizer: Tokenizer) -> torch.Tensor:
    out = ids.clone()
    out[out == -100] = tokenizer.pad_token_id
    return out


def print_batch_frm(
    batch: Dict[str, Any],
    split: Literal["train", "val", "test", "predict"],
    tokenizer: Tokenizer,
    dataloader_name: str = "",
) -> None:
    logger.info(
        f"Printing first entries of the tensors in batch for {split}/{dataloader_name}..."
    )
    print("input tokens:")
    _input_ids = _replace_100_with_pad(batch["input_ids"][0], tokenizer)
    print(tokenizer.decode(_input_ids))
    print("input_ids:")
    print(batch["input_ids"][0])
    if "clamp_mask" in batch:
        print("clamp_mask:")
        print(batch["clamp_mask"][0].int())
    if "target_ids" in batch and batch["target_ids"] is not None:
        print("target_ids:")
        print(batch["target_ids"][0])
        print("target tokens:")
        print(tokenizer.decode(_replace_100_with_pad(batch["target_ids"][0], tokenizer)))
