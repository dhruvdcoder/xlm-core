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
from typing import Callable, Dict, List, Literal, Optional, Any
from .types_mlm import MLMBatch
from xlm.utils.nn import pad_truncate_list
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

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
    input_ids = []
    target_ids = []
    attention_mask = []
    for i, example in enumerate(examples):
        temp = [bos_token_id] * add_bos + example + [eos_token_id] * add_eos
        _ids = torch.tensor(
            pad_truncate_list(temp, max_seq_len, pad_token_id, pad_left=False)
        )
        _attention_mask = torch.tensor(
            (
                [1] * len(_ids)
                if loss_on_padding
                else pad_truncate_list(
                    [1] * len(temp), max_seq_len, 0, pad_left=False
                )
            ),
            dtype=torch.bool,
        )
        input_ids.append(_ids)
        attention_mask.append(_attention_mask)
        _target_ids = _ids.clone()
        if not loss_on_padding:
            _target_ids[~_attention_mask] = -100

        target_ids.append(_target_ids)

    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    mask = torch.rand_like(input_ids) < t[:, None]
    if not loss_on_padding:
        mask = mask.logical_and(attention_mask)
    input_ids[mask] = mask_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": torch.stack(target_ids, dim=0),
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


def prepare_prefix_suffix_ids(
    prefix_ids: List[List[int]],
    suffix_ids: List[List[int]],
    pad_token_id: int,
    mask_token_id: int,
    eos_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    truncate: Literal["max", "block", None] = "block",
    loss_on_padding: bool = True,
) -> MLMBatch:
    """Prepare concatenated prefix and suffix ids for seq2seq tasks with padding on the right only

    Args:
        loss_on_padding: bool
          - If true, pad token is treated as a normal token: it has attention on it, it is predicted as a target token.
          - If false, it has no attention on it, it is not predicted as a target token (-100)
    """
    input_ids: List[TT] = []
    attention_mask: List[TT] = []
    target_ids: List[TT] = []
    add_eos = int(eos_token_id is not None)
    add_bos = int(
        bos_token_id is not None
    )  # always add bos before the suffix. Otherwise it is not needed.
    if truncate in ["max", None]:
        max_len = max(
            len(_prefix_ids) + len(_suffix_ids) + add_eos + add_bos
            for _prefix_ids, _suffix_ids in zip(prefix_ids, suffix_ids)
        )
        if truncate == "max" and max_seq_len is not None:
            max_len = max(max_len, max_seq_len)
    elif truncate == "block" and max_seq_len is not None:
        max_len = max_seq_len
    else:
        raise ValueError(f"Invalid truncate, max_seq_len: {max_seq_len}")
    assert max_len is not None
    batch_size = len(prefix_ids)
    t: Float[TT, " batch_size"] = torch.rand(batch_size)
    for i, (_prefix_ids, _suffix_ids) in enumerate(
        zip(prefix_ids, suffix_ids)
    ):
        # bos should not be masked
        suffix_mask = [0] * (len(_prefix_ids) + add_bos) + [1] * (len(_suffix_ids) + add_eos)
        temp = _prefix_ids + add_bos + _suffix_ids + add_eos
        if loss_on_padding:
            suffix_mask = pad_truncate_list(
                suffix_mask,
                max_len,
                1,
                pad_left=False,
            )
            temp = torch.tensor(
                    pad_truncate_list(
                    temp,
                    max_len,
                    pad_token_id,
                    pad_left=False,
                ),
                dtype=torch.long,
            )
            _input_ids = temp.clone()
            _mask = (torch.rand(len(_input_ids)) < t[i]).logical_and(
                torch.tensor(suffix_mask, dtype=torch.bool)
            )
            attention_mask.append(
                torch.tensor([1] * len(temp), dtype=torch.bool)
            )
            mask_len = (_mask==1).sum(dim=-1)
            _unmasked_inp = _input_ids[_mask==0]
            _masked_inp = torch.full((mask_len,), mask_token_id)
            _input_ids = torch.cat([_unmasked_inp, _masked_inp])
            input_ids.append(_input_ids)
            _target_ids = temp.clone()
            target_ids.append(_target_ids)
        else:
            temp = torch.tensor(temp, dtype=torch.long)
            _input_ids = temp.clone()
            _mask = (torch.rand(len(_input_ids)) < t[i]).logical_and(
                torch.tensor(suffix_mask, dtype=torch.bool)
            )
            attention_mask.append(
                torch.tensor(
                    pad_truncate_list(
                        [1]
                        * (
                            len(_prefix_ids)
                            + len(_suffix_ids)
                            + add_eos
                            + add_bos
                        ),
                        max_len,
                        0,
                        pad_left=False,
                    ),
                    dtype=torch.bool,
                )
            )          
            mask_len = (_mask==1).sum(dim=-1)
            _unmasked_inp = _input_ids[_mask==0]
            _masked_inp = torch.full((mask_len,), mask_token_id)
            _input_ids = torch.cat([_unmasked_inp, _masked_inp])
            _target_ids = temp.clone()
            pad_len = max(max_len - len(_input_ids), 0)
            if pad_len > 0:
                _padded_ids = torch.full((pad_len,), pad_token_id)
                _input_ids = torch.cat([_input_ids, _padded_ids])
                _target_ids = torch.cat([_target_ids, _padded_ids])
            input_ids.append(_input_ids)
            target_ids.append(_target_ids)
    target_ids = torch.stack(target_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    input_ids = torch.stack(input_ids, dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
    }


class DefaultMLMCollator(Collator):
    """Used for MLM pre-training with padded-truncated sequences.

    Batch:
        1. input_ids: Integer[TT, " batch seq_len"]: The input for the model with masks.
        2. attention_mask: Integer[TT, " batch seq_len"]: 1 for tokens that are not padding.
        3. target_ids: Integer[TT, " batch seq_len"]: The target ids to the model where the input if copied as is and masks are replaced with the correct token.

    Padding:
        - Padding is done on the right.
    """

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


class MLMSeq2SeqTrainCollator(Collator):
    """MLM training for seq2seq tasks.

    Batch:
        1. input_ids: Integer[TT, " batch seq_len"]: The input for the model with masks.
        2. attention_mask: Integer[TT, " batch seq_len"]: 1 for tokens that are not padding.
        3. target_ids: Integer[TT, " batch seq_len"]: The target ids to the model where the input if copied as is and masks are replaced with the correct token.

    Padding:
        - Padding is done on the right.
    """

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
        prefix_suffix = prepare_prefix_suffix_ids(
            [e["prompt_ids"] for e in examples],
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            self.tokenizer.mask_token_id,
            eos_token_id=self.tokenizer.eos_token_id if self.add_eos else None,
            bos_token_id=self.tokenizer.bos_token_id if self.add_bos else None,
            max_seq_len=(self.input_block_size + self.block_size),
            truncate=self.truncate,
            loss_on_padding=self.loss_on_padding,
        )
        return prefix_suffix


class MLMSeq2SeqPredCollator(Collator):
    """Input contains only the prefix and target_ids contain only the suffix if present."""

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
            [
                e["prompt_ids"]
                + [self.tokenizer.bos_token_id] * int(self.add_bos)
                for e in examples
            ],
            self.tokenizer.pad_token_id,
            max_seq_len=self.input_block_size,
        )
        # simply add eos and pad on right
        add_eos = int(self.add_eos)
        max_len = max(len(e["input_ids"]) + add_eos for e in examples)
        if self.block_size is not None and max_len > self.block_size:
            raise ValueError(
                f"Max length of target is greater than block size. {max_len} > {self.block_size}"
            )
        max_len = self.block_size
        target_ids = [
            pad_truncate_list(
                e["input_ids"]
                + [self.tokenizer.eos_token_id] * int(self.add_eos),
                max_len,
                self.tokenizer.pad_token_id,
                pad_left=False,
            )
            for e in examples
        ]

        return MLMBatch(
            input_ids=prefix["input_ids"],
            attention_mask=prefix["attention_mask"],
            target_ids=torch.tensor(target_ids, dtype=torch.long),
        )


def _replace_100_with_pad(ids: torch.Tensor, tokenizer: Tokenizer):
    _ids = ids.clone()
    _ids[_ids == -100] = tokenizer.pad_token_id
    return _ids


def print_batch_mlm(
    batch: Dict[str, Any],
    split: Literal["train", "val", "test", "predict"],
    tokenizer: Tokenizer,
    dataloader_name: str = "",
):
    """Print batch information for debugging MLM batches.

    Args:
        batch: The batch to print.
        split: The split name.
        tokenizer: The tokenizer to decode tokens.
        dataloader_name: Name of the dataloader.
    """
    logger.info(
        f"Printing first entries of the tensors in batch for {split}/{dataloader_name}..."
    )
    print("input tokens:")
    # replace -100 with <pad>
    _input_ids = _replace_100_with_pad(batch["input_ids"][0], tokenizer)
    print(tokenizer.decode(_input_ids))
    print("input_ids:")
    print(batch["input_ids"][0])
    print("attention_mask (int):")
    print(batch["attention_mask"][0].int())
    print("target_ids:")
    print(batch["target_ids"][0])
    print("target tokens:")
    _target_ids = _replace_100_with_pad(batch["target_ids"][0], tokenizer)
    print(tokenizer.decode(_target_ids))
