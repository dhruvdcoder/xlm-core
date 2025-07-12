import torch
from xlm.datamodule import (
    Collator,
    BaseCollatorInput,
    Seq2SeqCollatorInput,
    Tokenizer,
)
from jaxtyping import Float, Integer
from torch import Tensor as TT
from xlm.noise import NoiseSchedule
from typing import Callable, Dict, List, Literal, Optional
from .types_mlm import MLMBatch
from xlm.utils.nn import pad_truncate_list

################################################################################
# region: Collators


def mlm_single_segment_collate_fn(
    examples: List[List[int]],
    pad_token_id: int,
    mask_token_id: int,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    truncate: Literal["max", "block", None] = "block",
    loss_on_padding: bool = True,
    mask_all: bool = False,
) -> MLMBatch:
    # determine max_seq_len
    add_bos = int(bos_token_id is not None)
    add_eos = int(eos_token_id is not None)
    if truncate in ["max", None]:
        max_len_in_batch = max(len(v) for v in examples)
        if truncate == "max" and max_seq_len is not None:
            max_seq_len = max(max_len_in_batch, max_seq_len)
        else:
            max_seq_len = max_len_in_batch
    assert max_seq_len is not None
    batch_size = len(examples)
    t: Float[TT, " batch_size"] = (
        torch.rand(batch_size) if not mask_all else torch.ones(batch_size)
    )
    ids = []
    attention_mask = []
    for i, example in enumerate(examples):
        temp = [bos_token_id] * add_bos + example + [eos_token_id] * add_eos
        _ids = pad_truncate_list(
            temp, max_seq_len, pad_token_id, pad_left=False
        )
        _attention_mask = (
            [1] * len(_ids)
            if loss_on_padding
            else pad_truncate_list(
                [1] * len(temp), max_seq_len, 0, pad_left=False
            )
        )
        ids.append(_ids)
        attention_mask.append(_attention_mask)
    ids = torch.tensor(ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
    mask = torch.rand_like(ids) < t[:, None]
    input_ids = ids.clone()
    input_ids[mask] = mask_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": ids,
    }


def prepare_prefix_ids(
    prefix_ids: List[List[int]],
    pad_token_id: int,
    max_seq_len: Optional[int] = None,
    truncate: Literal["max", "block", None] = "block",
) -> Dict[str, TT]:
    """
    Prepare prefix ids for seq2seq tasks.

    Args:
        prefix_ids: List[List[int]]
        pad_token_id: int
        max_seq_len: Optional[int]
        truncate:
            - "max": Truncate to max(max_seq_len, max_in_batch).
                - when max_seq_len is not provided, it is the max in the batch.
            - "block": Pad-Truncate to max_seq_len.
            - None: Pad to max in the batch.

    Note: Prefixes if truncated will be truncated from the left.
    Returns:
        Dict[str, TT]:
            input_ids: Integer[TT, " batch seq_len"]
            attention_mask: Integer[TT, " batch seq_len"]
    """
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    if truncate in ["max", None]:
        max_len = max(len(_prefix_ids) for _prefix_ids in prefix_ids)
        if truncate == "max" and max_seq_len is not None:
            max_len = max(max_len, max_seq_len)
    elif truncate == "block" and max_seq_len is not None:
        max_len = max_seq_len
    else:
        raise ValueError(f"Invalid truncate, max_seq_len: {max_seq_len}")
    assert max_len is not None
    for _prefix_ids in prefix_ids:
        temp = _prefix_ids
        input_ids.append(
            pad_truncate_list(
                temp,
                max_len,
                pad_token_id,
                pad_left=True,
            )
        )
        attention_mask.append(
            pad_truncate_list([1] * len(temp), max_len, 0, pad_left=True)
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
    }


class DefaultMLMCollator(Collator):
    """Used for MLM pre-training with padded-truncated sequences."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        loss_on_padding: bool = True,
        truncate: Literal[
            "max", "block", None
        ] = "block",  # None is max in seq without a limit. max is max in seq upto block_size
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self._vocab_size = len(self.tokenizer)
        self.truncate = truncate
        self.loss_on_padding = loss_on_padding
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> MLMBatch:
        return mlm_single_segment_collate_fn(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            self.tokenizer.mask_token_id,
            bos_token_id=self.tokenizer.bos_token_id if self.add_bos else None,
            eos_token_id=self.tokenizer.eos_token_id if self.add_eos else None,
            max_seq_len=self.block_size,
            truncate=self.truncate,
            loss_on_padding=self.loss_on_padding,
        )


class MLMSeq2SeqCollator(Collator):
    """MLM training for seq2seq tasks."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        noise_schedule: NoiseSchedule,
        block_size: Optional[int] = None,
        input_block_size: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
        truncate: Literal[
            "max", "block", None
        ] = "block",  # None is max in seq without a limit. max is max in seq upto block_size
        loss_on_padding: bool = True,
    ):
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.block_size = block_size
        self.input_block_size = input_block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.truncate = truncate
        self.loss_on_padding = loss_on_padding

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> MLMBatch:
        prefix = prepare_prefix_ids(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            max_seq_len=self.input_block_size,
        )
        suffix = mlm_single_segment_collate_fn(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            self.tokenizer.mask_token_id,
            bos_token_id=self.tokenizer.bos_token_id if self.add_bos else None,
            eos_token_id=self.tokenizer.eos_token_id if self.add_eos else None,
            max_seq_len=self.block_size,
            truncate=self.truncate,
            loss_on_padding=self.loss_on_padding,
        )
        return MLMBatch(
            input_ids=torch.cat(
                [prefix["input_ids"], suffix["input_ids"]], dim=1
            ),
            attention_mask=torch.cat(
                [prefix["attention_mask"], suffix["attention_mask"]], dim=1
            ),
            target_ids=torch.cat(
                [prefix["input_ids"], suffix["target_ids"]], dim=1
            ),
        )


class MLMSeq2SeqPredCollator(MLMSeq2SeqCollator):
    """Masks all the suffix/target tokens and sends them in the target_ids of shape (batch_size, target_seq_len)"""

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> MLMBatch:
        prefix = prepare_prefix_ids(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            max_seq_len=self.input_block_size,
        )
        suffix = mlm_single_segment_collate_fn(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            self.tokenizer.mask_token_id,
            bos_token_id=self.tokenizer.bos_token_id if self.add_bos else None,
            eos_token_id=self.tokenizer.eos_token_id if self.add_eos else None,
            max_seq_len=self.block_size,
            truncate=self.truncate,
            loss_on_padding=self.loss_on_padding,
            mask_all=True,
        )

        return MLMBatch(
            input_ids=torch.cat(
                [prefix["input_ids"], suffix["input_ids"]], dim=1
            ),
            attention_mask=torch.cat(
                [prefix["attention_mask"], suffix["attention_mask"]], dim=1
            ),
            target_ids=torch.cat(
                [prefix["input_ids"], suffix["target_ids"]], dim=1
            ),
        )
