import torch
from xlm.datamodule import (
    Collator,
    BaseCollatorInput,
    Seq2SeqCollatorInput,
    Tokenizer,
)
from jaxtyping import Float, Integer
from torch.utils.data import IterableDataset
from torch import Tensor as TT
from .types_flexmdm import FlexMDMBatch
from typing import Callable, Dict, List, Literal, Optional, Any, Union
from .types_flexmdm import FlexMDMBatch
from .noise_flexmdm import FlexMDMNoiseSchedule
from xlm.utils.nn import pad_truncate_list
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

################################################################################
# region: Collators


class FlexMDMEmptyDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        num_examples: int,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.max_length = max_length

    def __iter__(self):
        for _ in range(self.num_examples):
            ex = {"input_ids": []}
            yield ex


def sample_time(batch_size: int, device: torch.device) -> torch.Tensor:
    eps = 1e-6
    interval = 1.0 - eps
    interval_size = interval / batch_size
    u = torch.rand(batch_size, device=device)
    return (
        torch.arange(batch_size, device=device, dtype=u.dtype) + u
    ) * interval_size


def flexmdm_single_segment_collate_fn(
    examples: List[List[int]],
    noise_schedule: FlexMDMNoiseSchedule,
    pad_token_id: int,
    mask_token_id: int,
    # bos_token_id: int,  # TODO remove arg
    eos_token_id: int,
    max_seq_len: Optional[int] = None,
    truncate: Literal["max", "block", None] = "max",
    mask_all: bool = False,
) -> FlexMDMBatch:

    assert eos_token_id is not None

    # determine max_seq_len
    if truncate in ["max", None]:
        max_len_in_batch = max(len(v) for v in examples)
        if truncate == "max" and max_seq_len is not None:
            max_seq_len = max(max_len_in_batch, max_seq_len) + 1  # +1 for bos
        else:
            max_seq_len = max_len_in_batch
    assert max_seq_len is not None
    batch_size = len(examples)

    t = sample_time(batch_size, device="cpu")
    # insertion_schedule, unmasking_schedule = LinearSchedule(), LinearSchedule()

    x1 = []
    for i, example in enumerate(examples):
        if len(example) + 1 > max_seq_len and truncate != "block":
            raise ValueError(
                f"Sequence length {len(example) + 1} is greater than max_seq_len {max_seq_len}"
            )
        x1.append(
            torch.tensor(
                pad_truncate_list(
                    # [bos_token_id] + example,
                    example + [eos_token_id],
                    max_seq_len,
                    pad_token_id,
                    pad_left=False,
                )
            )
        )

    x1 = torch.stack(x1, dim=0)
    xt = x1.clone()
    not_eos = xt != eos_token_id

    eps = 1e-6
    insertion_time = noise_schedule.insertion_noise_schedule.sample(
        (batch_size, max_seq_len), device=t.device
    )
    insertion_time = eps + (1 - eps) * insertion_time  # ensure t1 is not zero
    unmasking_time = noise_schedule.unmasking_noise_schedule.sample_truncated(
        insertion_time, (batch_size, max_seq_len), device=t.device
    )

    clean_tokens = x1.ne(pad_token_id)
    deleted_tokens = clean_tokens & (t[:, None] < insertion_time) & not_eos
    masked_tokens = (
        clean_tokens
        & (t[:, None] >= insertion_time)
        & (t[:, None] < unmasking_time)
        & not_eos
    )

    xt = torch.where(
        deleted_tokens,
        pad_token_id,  # for deletion, change to pad token
        torch.where(
            masked_tokens,
            mask_token_id,  # for masking, change to mask token
            x1,
        ),
    )

    # st: original positions of the non-deleted tokens
    st = xt.ne(pad_token_id).argsort(dim=1, descending=True, stable=True)
    xt = torch.gather(xt, 1, st)  # squeeze together the non-deleted tokens
    st[xt == pad_token_id] = 0

    x1_len = (x1 != pad_token_id).sum(dim=1)
    xt_len = (xt != pad_token_id).sum(dim=1)

    # gaps: will hold gap length for the gap that is left of the non-deleted token
    temp = st.clone()

    pad_front = (
        temp.new_zeros((temp.shape[0], 1)) - 1
    )  # -1 for the front padding
    # don't need pad_back because EOS is added at the end and never deleted
    # pad_back = temp.new_zeros((temp.shape[0], 1))
    # temp = torch.cat([pad_front, temp, pad_back], dim=1)  # Add a leading zero
    # temp.scatter_(
    #    1, xt_len.unsqueeze(1) + 1, x1_len.unsqueeze(1)
    # )  # Fill the last position with x1_len
    temp = torch.cat([pad_front, temp], dim=1)  # Add a leading zero

    gaps = temp[:, 1:] - temp[:, :-1] - 1
    gaps = torch.clamp(gaps, min=0)

    idx = torch.arange(gaps.size(1), device=xt.device).unsqueeze(
        0
    )  # shape [1, max_gap]
    # mask = idx <= xt_len.unsqueeze(1)
    mask = idx < xt_len.unsqueeze(1)
    gaps[~mask] = 0

    ret = {
        "input_ids": xt,
        "input_positions": st,
        "gaps": gaps,
        "gaps_mask": mask,
        "target_ids": x1,
        "token_weight": noise_schedule.unmasking_noise_schedule.rate_scale_factor(
            t
        ),
        "length_weight": noise_schedule.insertion_noise_schedule.rate_scale_factor(
            t
        ),
        "t": t,
        "max_length": max_seq_len,
    }
    return ret


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
    noise_schedule: FlexMDMNoiseSchedule,
    pad_token_id: int,
    mask_token_id: int,
    eos_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    truncate: Literal["max", "block", None] = "block",
    loss_on_padding: bool = True,
    bos_location: Literal["before_prefix", "after_prefix"] = "after_prefix",
) -> FlexMDMBatch:
    """Prepare concatenated prefix and suffix ids for seq2seq tasks with padding on the right only

    Args:
        loss_on_padding: bool
          - If true, pad token is treated as a normal token: it has attention on it, it is predicted as a target token.
          - If false, it has no attention on it, it is not predicted as a target token (-100)
    """
    input_ids: List[TT] = []
    attention_mask: List[TT] = []
    target_ids: List[TT] = []
    mask: List[TT] = []
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
    t: Float[TT, " batch_size"] = noise_schedule.sample_t(batch_size)
    noise_rate, total_noise = noise_schedule(t)
    for i, (_prefix_ids, _suffix_ids) in enumerate(
        zip(prefix_ids, suffix_ids)
    ):
        # bos should not be masked
        suffix_mask = pad_truncate_list(
            [0] * (len(_prefix_ids) + add_bos)
            + [1] * (len(_suffix_ids) + add_eos),
            max_len,
            1,
            pad_left=False,
        )
        temp = pad_truncate_list(
            (
                _prefix_ids + ([bos_token_id] * add_bos)
                if bos_location == "after_prefix"
                else (([bos_token_id] * add_bos) + _prefix_ids)
            )
            + _suffix_ids
            + ([eos_token_id] * add_eos),
            max_len,
            pad_token_id,
            pad_left=False,
        )
        _mask = (
            torch.rand(len(temp), device=total_noise.device)
            < -(torch.expm1(-total_noise[i].unsqueeze(-1)))
        ).logical_and(torch.tensor(suffix_mask, dtype=torch.bool))
        _input_ids = torch.tensor(temp, dtype=torch.long)
        input_ids.append(_input_ids)
        if loss_on_padding:
            attention_mask.append(
                torch.tensor([1] * len(temp), dtype=torch.bool)
            )
            target_ids.append(_input_ids.clone())
            mask.append(_mask)
        else:
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
            mask.append(
                _mask.logical_and(attention_mask[-1])
            )  # no input masks in padding
            _target_ids = _input_ids.clone()
            _target_ids[~attention_mask[-1]] = -100  # no loss on padding
            target_ids.append(_target_ids)
    target_ids = torch.stack(target_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    input_ids = torch.stack(input_ids, dim=0)
    mask = torch.stack(mask, dim=0)
    input_ids[mask] = mask_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
        "noise_rate": noise_rate,
        "total_noise": total_noise,
        "t": t,
    }


class DefaultFlexMDMCollator(Collator):
    """Used for FlexMDM pre-training with padded-truncated sequences.

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
        noise_schedule: FlexMDMNoiseSchedule,
        truncate: Literal[
            "max", "block", None
        ] = "block",  # None is max in seq without a limit. max is max in seq upto block_size
        add_bos: bool = True,
    ):
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self._vocab_size = len(self.tokenizer)
        self.truncate = truncate
        self.add_bos = add_bos

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> FlexMDMBatch:
        return flexmdm_single_segment_collate_fn(
            [e["input_ids"] for e in examples],
            self.noise_schedule,
            self.tokenizer.pad_token_id,
            self.tokenizer.mask_token_id,
            # bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=self.block_size,
            truncate=self.truncate,
        )


class DefaultFlexMDMUnconditionalPredCollator(Collator):
    """For unconditional prediction."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> FlexMDMBatch:
        batch = {
            "input_ids": torch.cat(
                [
                    torch.full(
                        (len(examples), 1),
                        # self.tokenizer.bos_token_id,
                        # self.tokenizer.pad_token_id,
                        self.tokenizer.eos_token_id,
                        dtype=torch.int64,
                        device="cpu",
                    ),
                    torch.full(
                        (len(examples), self.block_size - 1),
                        self.tokenizer.pad_token_id,
                        dtype=torch.int64,
                        device="cpu",
                    ),
                ],
                dim=-1,
            )
        }
        return batch


class FlexMDMSeq2SeqTrainCollator(Collator):
    """FlexMDM training for seq2seq tasks.

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
        noise_schedule: FlexMDMNoiseSchedule,
        block_size: Optional[int] = None,
        input_block_size: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
        truncate: Literal[
            "max", "block", None
        ] = "block",  # None is max in seq without a limit. max is max in seq upto block_size
        loss_on_padding: bool = True,
        bos_location: Literal[
            "before_prefix", "after_prefix"
        ] = "after_prefix",
    ):
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.block_size = block_size
        self.input_block_size = input_block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.truncate = truncate
        self.loss_on_padding = loss_on_padding
        self.bos_location = bos_location

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> FlexMDMBatch:
        prefix_suffix = prepare_prefix_suffix_ids(
            [e["prompt_ids"] for e in examples],
            [e["input_ids"] for e in examples],
            self.noise_schedule,
            self.tokenizer.pad_token_id,
            self.tokenizer.mask_token_id,
            eos_token_id=self.tokenizer.eos_token_id if self.add_eos else None,
            bos_token_id=self.tokenizer.bos_token_id if self.add_bos else None,
            max_seq_len=(self.input_block_size + self.block_size),
            truncate=self.truncate,
            loss_on_padding=self.loss_on_padding,
            bos_location=self.bos_location,
        )
        return prefix_suffix


class FlexMDMSeq2SeqPredCollator(Collator):
    """Input contains only the prefix and target_ids contain only the suffix if present.

    How is this different from FlexMDMSeq2SeqTrainCollator?
        -  FlexMDMSeq2SeqTrainCollator's input_ids contain the joined sequence and target_ids also contain the target for the whole sequence. But FlexMDMSeq2SeqPredCollator's input_ids contain only the prefix and target_ids contain only the suffix if present.

    Batch:
        1. input_ids: Integer[TT, " batch seq_len"]: Input contains only the prefix
        2. attention_mask: Integer[TT, " batch seq_len"]: 1 for tokens that are not padding.
        3. target_ids: Integer[TT, " batch seq_len"]: Target contains only the suffix if present.
        4. noise_rate: Float[TT, " batch"]: The noise rate for the model.
        5. total_noise: Float[TT, " batch"]: The total noise for the model.
        6. t: Float[TT, " batch"]: The time step for the model.

    Padding:
        - There is padding on both sides because all prefixes end at the same position.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        noise_schedule: FlexMDMNoiseSchedule,
        block_size: Optional[int] = None,
        input_block_size: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
        truncate: Literal[
            "max", "block", None
        ] = "block",  # None is max in seq without a limit. max is max in seq upto block_size
        loss_on_padding: bool = True,
        bos_location: Literal[
            "before_prefix", "after_prefix"
        ] = "after_prefix",
    ):
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.block_size = block_size
        self.input_block_size = input_block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.truncate = truncate
        self.loss_on_padding = loss_on_padding
        self.bos_location = bos_location

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> FlexMDMBatch:

        prefix = prepare_prefix_ids(
            [
                (
                    e["prompt_ids"]
                    + [self.tokenizer.bos_token_id] * int(self.add_bos)
                    if self.bos_location == "after_prefix"
                    else (
                        [self.tokenizer.bos_token_id] * int(self.add_bos)
                        + e["prompt_ids"]
                    )
                )
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

        return FlexMDMBatch(
            input_ids=prefix["input_ids"],
            attention_mask=prefix["attention_mask"],
            target_ids=torch.tensor(target_ids, dtype=torch.long),
            noise_rate=None,
            total_noise=None,
            t=None,
        )


# endregion: Collators
################################################################################


################################################################################
# region: Generalized Collator


def flexmdm_train_collate_fn(
    prompt_ids: Optional[List[List[int]]],
    target_ids: List[List[int]],
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    max_seq_len: Optional[int] = None,
) -> Dict[str, TT]:
    """
    Unconditional case:
        Input:
            - prompt_ids = None
            - target_ids = [The quick brown fox jumped over the lazy dog]
        Output:
            - ids   = [The quick brown fox jumped over the lazy dog] [eos] [pad pad pad]
            - fixed = [ 0    0     0     0    0    0    0   0    0     1     0   0   0 ]

    Conditional case:
        Input:
            - prompt_ids = [The quick brown fox]
            - target_ids = [jumped over the lazy dog]
        Output:
            - ids   = [The quick brown fox] [bos] [jumped over the lazy dog] [eos] [pad pad pad]
            - fixed = [ 1    1     1    1     1      0     0    0    0   0     1     0   0   0 ]
    """

    # start with prompt ids, if any, followed by bos
    if prompt_ids is not None:
        ids = [_prompt_ids + [bos_token_id] for _prompt_ids in prompt_ids]
        fixed = [[1] * (len(_prompt_ids) + 1) for _prompt_ids in prompt_ids]
    else:
        ids = [[] for _ in target_ids]
        fixed = [[] for _ in target_ids]

    # add targets
    ids = [_ids + target_ids[i] for i, _ids in enumerate(ids)]
    fixed = [
        _fixed + [0] * len(target_ids[i]) for i, _fixed in enumerate(fixed)
    ]

    assert max_seq_len is not None

    # truncate, add eos, pad
    for i, _ids in enumerate(ids):
        ids[i] = (
            _ids[: max_seq_len - 1]
            + [eos_token_id]
            + [pad_token_id] * (max_seq_len - len(_ids) - 1)
        )
        fixed[i] = (
            fixed[i][: max_seq_len - 1]
            + [1]
            + [0] * (max_seq_len - len(fixed[i]) - 1)
        )

    ids = torch.tensor(ids)
    fixed = torch.tensor(fixed)

    ret = {"input_ids": ids, "fixed": fixed}
    return ret


class FlexMDMTrainCollator(Collator):

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        input_block_size: Optional[int] = 0,
    ):
        self.block_size = block_size
        self.input_block_size = input_block_size
        self.tokenizer = tokenizer
        try:
            self._vocab_size = len(self.tokenizer)
        except TypeError:
            self._vocab_size = self.tokenizer.full_vocab_size

    def __call__(
        self,
        examples: List[Union[BaseCollatorInput, Seq2SeqCollatorInput]],
    ) -> Dict[str, TT]:
        return flexmdm_train_collate_fn(
            prompt_ids=(
                [e["prompt_ids"] for e in examples]
                if "prompt_ids" in examples[0]
                else None
            ),
            target_ids=[e["input_ids"] for e in examples],
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=(self.input_block_size + self.block_size),
        )


def flexmdm_pred_collate_fn(
    num_examples: int,
    prompt_ids: Optional[List[List[int]]],
    target_ids: List[List[int]],
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    max_seq_len: Optional[int] = None,
) -> Dict[str, Optional[TT]]:

    if prompt_ids is not None:  # conditional case: start with [prefix] [bos]
        ids = [_prompt_ids + [bos_token_id] for _prompt_ids in prompt_ids]
        fixed_gaps = [
            [1] * (len(_prompt_ids) + 1) for _prompt_ids in prompt_ids
        ]
    else:  # unconditional case: start with empty
        ids = [[] for _ in range(num_examples)]
        fixed_gaps = [[] for _ in range(num_examples)]

    assert max_seq_len is not None

    # truncate
    for i, _ids in enumerate(ids):
        ids[i] = _ids[: max_seq_len - 1]
        fixed_gaps[i] = fixed_gaps[i][: max_seq_len - 1]

    # add eos, pad
    for i, _ids in enumerate(ids):
        ids[i] = (
            _ids
            + [eos_token_id]
            + [pad_token_id] * (max_seq_len - len(_ids) - 1)
        )
        fixed_gaps[i] = (
            fixed_gaps[i] + [0] + [0] * (max_seq_len - len(fixed_gaps[i]) - 1)
        )

    ids = torch.tensor(ids)
    fixed_gaps = torch.tensor(fixed_gaps)

    # -- add eos, pad target ids (not needed for pred but required for seq2seq logging)  # TODO why is this necessary?
    if target_ids is not None:
        for i, _target_ids in enumerate(target_ids):
            target_ids[i] = (
                _target_ids
                + [eos_token_id]
                + [pad_token_id] * (max_seq_len - len(_target_ids) - 1)
            )
        target_ids = torch.tensor(target_ids)
    # --

    ret = {"input_ids": ids, "fixed": fixed_gaps, "target_ids": target_ids}
    return ret


class FlexMDMPredCollator(Collator):
    """
    Unconditional case:
        Input:
            - prompt_ids = None
        Output:
            - ids          = [eos] [pad pad pad]
            - fixed[_gaps] = [ 0     1   1   1 ]

    Conditional case:
        Input:
            - prompt_ids = [The quick brown fox]
        Output:
            - ids          = [The quick brown fox] [bos] [eos] [pad pad pad]
            - fixed[_gaps] = [ 1    1     1    1     1     0     1   1   1 ]
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        input_block_size: Optional[int] = 0,
    ):
        self.block_size = block_size
        self.input_block_size = input_block_size
        self.tokenizer = tokenizer
        try:
            self._vocab_size = len(self.tokenizer)
        except TypeError:
            self._vocab_size = self.tokenizer.full_vocab_size

    def __call__(
        self,
        examples: List[Union[BaseCollatorInput, Seq2SeqCollatorInput]],
    ) -> Dict[str, TT]:
        has_prefix = "prompt_ids" in examples[0]
        return flexmdm_pred_collate_fn(
            num_examples=len(examples),
            prompt_ids=(
                [e["prompt_ids"] for e in examples] if has_prefix else None
            ),
            target_ids=[e["input_ids"] for e in examples],
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=(
                (self.input_block_size + self.block_size)
                if has_prefix
                else self.block_size
            ),
        )


# endregion: Generalized Collators
################################################################################


def print_batch_flexmdm(
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
    print(tokenizer.decode(batch["input_ids"][0]))
    print("fixed positions/gaps:")
    print(batch["fixed"][0])
