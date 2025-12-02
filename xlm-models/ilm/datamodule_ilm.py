from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Any,
    Optional,
)
from torch import Tensor as TT
from torch.utils.data import IterableDataset
import numpy as np
import torch
from xlm.noise import NoiseSchedule
from xlm.utils.imports import get_function
from xlm.utils.rank_zero import RankedLogger
from xlm.datamodule import (
    Seq2SeqCollatorInput,
    Tokenizer,
    Collator,
    BaseCollatorInput,
)
from .types_ilm import ILMBatch
from xlm.utils.nn import pad_truncate_list

logger = RankedLogger(__name__, rank_zero_only=True)


class ILMEmptyDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        num_examples: int,
    ):
        """
        Args:
            tokenizer_kwargs: Keyword arguments for the tokenizer.

            empty_text: For MLM, you will want to set the `empty_text` to a sequence of all mask tokens.
        """
        self.tokenizer = tokenizer
        self.num_examples = num_examples

    def __iter__(self):
        for _ in range(self.num_examples):
            ex = self.tokenizer(
                "",
                add_special_tokens=False,
            )
            yield ex


################################################################################
# region: Collators


def _drop_uniformly(seq_len: int, n_drops: int) -> List[int]:
    return np.random.permutation(seq_len)[
        :n_drops
    ]  # np.random.permutation is the fastest


def _n_drop_all(seq_len: int) -> int:
    return seq_len


def _n_drop_uniformly(seq_len: int) -> int:
    return np.random.randint(seq_len + 1)


def ilm_drop_fn(
    segment_input_ids: List[
        int
    ],  # expected to be the segment only an not the whole sequence
    bos_token_id: int,
    cls_token_id: Optional[int],
    global_offset: int = 0,  # support have multiple target segments with some fixed segments in between
    sample_n_drops_fn: Callable[[int], int] = _n_drop_uniformly,
    drop_indices_fn: Callable[[int, int], List[int]] = _drop_uniformly,
) -> Dict[str, List[int]]:
    """Drops tokens from a single segment of a single sequence. Adds bos. Adds cls as requested."""
    _input_ids = segment_input_ids
    offset = 2 if cls_token_id is not None else 1
    target_seq_indices: List[int] = []
    target_vocab_indices: List[int] = []
    target_values: List[int] = []
    n_drops_seq_indices: List[int] = []
    n_drops_values: List[int] = []
    _orig_seq_len = len(segment_input_ids)
    _n_drops = int(sample_n_drops_fn(_orig_seq_len))
    _drop_indices = drop_indices_fn(
        _orig_seq_len, _n_drops
    )  # indices of the dropped tokens in the original sequence
    _drop_indices_set = set(_drop_indices)  # in the original sequence
    # TODO: maybe pre-allocate np.arrays?

    _input_ids_with_drops: List[int] = (
        [cls_token_id, bos_token_id] + [None] * (_orig_seq_len - _n_drops)
        if cls_token_id is not None
        else [bos_token_id] + [None] * (_orig_seq_len - _n_drops)
    )
    _prev_n_drops = 0
    i = offset - 1  # index in the post-drop sequence
    ############################################################
    # j, i move like so:
    # c  b  w1 w2 w3 w4 w5 w6
    # 0  1  2  3  4  5  6  7
    #       x  x     x
    # offset = 2, _orig_seq_len = 6, _n_drops = 3
    # (j, i) = (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 4)
    # res: c b w3 w5 w7
    ############################################################
    # the loop is over the original sequence length without the special tokens
    for j in range(_orig_seq_len):
        if j in _drop_indices_set:
            target_seq_indices.append(global_offset + i)  # prepared globally
            target_vocab_indices.append(int(_input_ids[j]))
            target_values.append(1)
            n_drops_seq_indices.append(global_offset + i)  # prepared globally
            n_drops_values.append(1)
            _prev_n_drops += 1
        else:
            # encountered the next non-dropped position
            _input_ids_with_drops[i + 1] = _input_ids[j]
            # n_drops_seq_indices.append(global_offset + i)  # prepared globally
            # n_drops_values.append(_prev_n_drops)
            # reset
            _prev_n_drops = 0
            i += 1
    # checks
    assert not any(_idx is None for _idx in _input_ids_with_drops)
    return {
        "segment_input_ids_with_drops": _input_ids_with_drops,
        "segment_target_seq_indices": target_seq_indices,  # global
        "segment_target_vocab_indices": target_vocab_indices,
        "segment_target_values": target_values,
        "segment_n_drops_seq_indices": n_drops_seq_indices,  # global
        "segment_n_drops_values": n_drops_values,
    }


def prepare_prefix_ids(
    prefix_ids: List[List[int]],
    pad_token_id: int,
    max_seq_len: Optional[int] = None,
    cls_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    bos_side: Literal["left", "right"] = "right",
) -> Dict[str, TT]:
    add_cls = 1 if cls_token_id is not None else 0
    add_bos = 1 if bos_token_id is not None else 0
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    token_type_ids: List[List[int]] = []
    cls_positions: List[int] = []
    if max_seq_len is None:
        max_len = max(
            len(_prefix_ids) + add_cls + add_bos for _prefix_ids in prefix_ids
        )
    else:
        max_len = max_seq_len
    for _prefix_ids in prefix_ids:
        if bos_side == "right":
            temp = (
                ([cls_token_id] if add_cls else [])
                + _prefix_ids
                + ([bos_token_id] if add_bos else [])
            )
        else:
            temp = (
                ([cls_token_id] if add_cls else [])
                + ([bos_token_id] if add_bos else [])
                + _prefix_ids
            )
        _padded, num_padded = pad_truncate_list(
            temp,
            max_len,
            pad_token_id,
            pad_left=True,
            return_num_padded=True,
        )
        input_ids.append(_padded)
        cls_positions.append(num_padded)
        attention_mask.append(
            pad_truncate_list([1] * len(temp), max_len, 0, pad_left=True)
        )
        token_type_ids.append(
            pad_truncate_list(
                ([0] if add_cls else [])
                + [1] * (len(temp) - add_cls - add_bos)
                + ([2] if add_bos else []),
                max_len,
                -1,
                pad_left=True,
            )
        )
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "cls_position": torch.tensor(cls_positions, dtype=torch.long),
    }


def ilm_single_segment_collate_target_fn(
    examples: List[List[int]],  # List[Dict[Literal["input_ids"], List[int]]],
    pad_token_id: int,
    bos_token_id: int,
    vocab_size: int,
    cls_token_id: Optional[int],
    type_extension_id: int = 2,
    pad_left: bool = False,
    max_seq_len: Optional[
        int
    ] = None,  # provide this only if you want to pad_truncate to a fixed length
    truncate: Literal["max", "block", None] = "block",
    global_offset: int = 0,  # support have multiple target segments will with some fixed segments in between
    return_dense_target: bool = False,
    return_dense_n_drops: bool = True,
    drop_indices_fn: Callable[[int, int], List[int]] = _drop_uniformly,
    sample_n_drops_fn: Callable[[int], int] = _n_drop_uniformly,
) -> ILMBatch:
    # We assume that there is no prefix and any token can be dropped. If there is prefix, it should be separated before the sequence is sent here.
    # Flow 1:
    # 1. Perform the dropping and construct the target_ids, n_drops, attention_mask, and token_type_ids.
    # 2. Add the special tokens: CLS and BOS to the left of input_ids, move the target_ids, n_drops, attention_mask, and token_type_ids accordingly.
    # Flow 2:
    # 1. Add the special tokens: CLS and BOS to the left of input_ids.
    # 2. Perform the dropping (protect the special tokens), and construct the target_ids, n_drops, attention_mask, and token_type_ids.
    # For seq2seq tasks, we won't add cls here.
    target_batch_indices: List[int] = []
    target_seq_indices: List[int] = []
    target_vocab_indices: List[int] = []
    target_values: List[int] = []
    n_drops_batch_indices: List[int] = []
    n_drops_seq_indices: List[int] = []
    n_drops_values: List[int] = []
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    token_type_ids: List[List[int]] = []
    max_seq_len_in_batch = 0
    results: List[Dict[str, Any]] = []
    # two passes are needed to figure out the max_seq_len
    for e, _example in enumerate(examples):
        single_seq_drop_result = ilm_drop_fn(
            _example,
            bos_token_id,
            cls_token_id,
            global_offset,
            sample_n_drops_fn=sample_n_drops_fn,
            drop_indices_fn=drop_indices_fn,
        )
        # add batch_index
        single_seq_drop_result["segment_target_batch_indices"] = [e] * len(
            single_seq_drop_result["segment_target_seq_indices"]
        )
        single_seq_drop_result["segment_n_drops_batch_indices"] = [e] * len(
            single_seq_drop_result["segment_n_drops_seq_indices"]
        )
        max_seq_len_in_batch = max(
            max_seq_len_in_batch,
            len(single_seq_drop_result["segment_input_ids_with_drops"]),
        )
        results.append(single_seq_drop_result)

    # compute max post-drop
    if truncate == "max":  # max in batch
        max_len = max_seq_len_in_batch
    elif truncate == "block":  # max in block
        if max_seq_len is None:
            raise ValueError("")
        max_len = max_seq_len
    elif truncate is None:  # no truncation
        max_len = max_seq_len_in_batch
    else:
        raise ValueError(f"Invalid truncate value: {truncate}")

    for single_seq_drop_result in results:
        # store the results for sparse tensors
        target_batch_indices.extend(
            single_seq_drop_result["segment_target_batch_indices"]
        )
        target_seq_indices.extend(
            single_seq_drop_result["segment_target_seq_indices"]
        )
        target_vocab_indices.extend(
            single_seq_drop_result["segment_target_vocab_indices"]
        )
        target_values.extend(single_seq_drop_result["segment_target_values"])
        n_drops_batch_indices.extend(
            single_seq_drop_result["segment_n_drops_batch_indices"]
        )
        n_drops_seq_indices.extend(
            single_seq_drop_result["segment_n_drops_seq_indices"]
        )
        n_drops_values.extend(single_seq_drop_result["segment_n_drops_values"])

        # do the padding and prepare the final collated batch
        input_ids.append(
            pad_truncate_list(
                single_seq_drop_result["segment_input_ids_with_drops"],
                max_len,
                pad_token_id,
                pad_left,
            )
        )
        attention_mask.append(
            pad_truncate_list(
                [1]
                * len(single_seq_drop_result["segment_input_ids_with_drops"]),
                max_len,
                0,
                pad_left,
            )
        )
        # type_ids: 0 for CLS, 1 for prefix (fixed), 2 for non-prefix (not fixed) tokens including BOS
        # we don't have prefix in this function
        token_type_ids.append(
            pad_truncate_list(
                (
                    [0, 2]
                    + 
                    (
                        [type_extension_id]
                        * (
                            len(
                                single_seq_drop_result[
                                    "segment_input_ids_with_drops"
                                ]
                            )
                            - 2
                        )
                        if cls_token_id is not None
                        else [type_extension_id]
                        * (
                            len(
                                single_seq_drop_result[
                                    "segment_input_ids_with_drops"
                                ]
                            )
                            - 1
                        )
                    )
                ),
                max_len,
                type_extension_id,
                pad_left,
            )
        )
    # checks
    assert len(input_ids[0]) == max_len
    assert (
        len(input_ids[0]) == len(attention_mask[0]) == len(token_type_ids[0])
    )
    # create the sparse tensors
    target_ids = torch.sparse_coo_tensor(
        indices=[  # type: ignore
            target_batch_indices,
            target_seq_indices,
            target_vocab_indices,
        ],
        values=target_values,
        size=(len(examples), global_offset + max_len, vocab_size),
        check_invariants=False,
        is_coalesced=False,
    )
    n_drops_counts = torch.sparse_coo_tensor(
        indices=[  # type: ignore
            n_drops_batch_indices,
            n_drops_seq_indices,
        ],
        values=n_drops_values,
        size=(len(examples), global_offset + max_len),
        check_invariants=False,
        is_coalesced=False,
    )
    # checks
    assert (n_drops_counts.to_dense() == target_ids.to_dense().sum(-1)).all()
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "target_ids": (
            target_ids.to_dense() if return_dense_target else target_ids
        ),
        "n_drops": (
            n_drops_counts.to_dense()
            if return_dense_n_drops
            else n_drops_counts
        ),
        "constraint": None,
        "cls_position": None,
        "target_attention_mask": None,
    }


def prepare_target_ids_for_test(
    target_ids: List[List[int]],
    pad_token_id: int,
    bos_token_id: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> Dict[str, TT]:
    add_bos = 1 if bos_token_id is not None else 0
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    token_type_ids: List[List[int]] = []
    if max_seq_len is None:
        max_len = max(len(_target_ids) + add_bos for _target_ids in target_ids)
    else:
        max_len = max_seq_len
    for _target_ids in target_ids:
        temp = ([bos_token_id] if add_bos else []) + _target_ids
        input_ids.append(
            pad_truncate_list(
                temp,
                max_len,
                pad_token_id,
                pad_left=False,
            )
        )
        attention_mask.append(
            pad_truncate_list([1] * len(temp), max_len, 0, pad_left=False)
        )
        token_type_ids.append(
            pad_truncate_list(
                ([0] if add_bos else []) + [1] * (len(temp) - add_bos),
                max_len,
                2,
                pad_left=False,
            )
        )
    return {
        "target_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
    }


class DefaultILMCollator(Collator):
    """Used for pre-training."""

    sample_n_drops_fn: Callable[[int], int] = _n_drop_uniformly
    drop_indices_fn: Callable[[int, int], List[int]] = _drop_uniformly

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        loss_on_padding: bool = False,
        return_dense_target: bool = False,  # setting to two will increase the cpu memory usage
        truncate: Literal["max", "block", None] = "block",
    ):
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self._vocab_size = len(self.tokenizer)
        assert loss_on_padding is False, "Loss on padding is not supported"
        self.return_dense_target = return_dense_target
        self.truncate = truncate

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not set")
            self._vocab_size = len(self.tokenizer)
        return self._vocab_size

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> ILMBatch:
        batch = ilm_single_segment_collate_target_fn(
            [e["input_ids"] for e in examples],  # type: ignore
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.vocab_size,
            self.tokenizer.cls_token_id,
            type_extension_id=2,  # there is not prefix
            pad_left=False,
            max_seq_len=self.block_size,
            truncate=self.truncate,
            global_offset=0,
            return_dense_target=self.return_dense_target,
            return_dense_n_drops=True,
            sample_n_drops_fn=self.__class__.sample_n_drops_fn,
            drop_indices_fn=self.__class__.drop_indices_fn,
        )
        batch["cls_position"] = torch.zeros(
            batch["input_ids"].shape[0], dtype=torch.long
        )
        return batch


class ILMSeq2SeqCollator:
    """Drops tokens from the suffix only."""

    sample_n_drops_fn: Callable[[int], int] = _n_drop_uniformly
    drop_indices_fn: Callable[[int, int], List[int]] = _drop_uniformly

    def __init__(
        self,
        tokenizer: Tokenizer,
        noise_schedule: NoiseSchedule,
        block_size: Optional[int] = None,
        input_block_size: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.input_block_size = input_block_size
        self._vocab_size = (
            len(self.tokenizer) if self.tokenizer is not None else None
        )

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not set")
            self._vocab_size = len(self.tokenizer)
        return self._vocab_size

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> ILMBatch:
        # handle the prefix
        cls_token_id = self.tokenizer.cls_token_id
        prefix = prepare_prefix_ids(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            max_seq_len=self.input_block_size,
            cls_token_id=cls_token_id,
        )
        global_offset = prefix["input_ids"].shape[1]

        suffix = ilm_single_segment_collate_target_fn(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.vocab_size,
            cls_token_id=None,  # we already added to prefix
            type_extension_id=2,  # -1 for left-pad, 0 for CLS, 1 for prefix/fixed, 2 for non-prefix/not fixed
            pad_left=False,
            max_seq_len=self.block_size,
            global_offset=global_offset,
            return_dense_target=False,
            return_dense_n_drops=True,
            sample_n_drops_fn=self.__class__.sample_n_drops_fn,
            drop_indices_fn=self.__class__.drop_indices_fn,
        )
        # cat prefix and suffix
        return {
            "input_ids": torch.cat(
                [prefix["input_ids"], suffix["input_ids"]], dim=1
            ),
            "attention_mask": torch.cat(
                [prefix["attention_mask"], suffix["attention_mask"]], dim=1
            ),
            "token_type_ids": torch.cat(
                [prefix["token_type_ids"], suffix["token_type_ids"]], dim=1
            ),
            "target_ids": suffix["target_ids"],
            "n_drops": suffix["n_drops"],
            "constraint": None,
            "cls_position": prefix["cls_position"],
            "target_attention_mask": None,
        }


class ILMSeq2SeqPredCollator(ILMSeq2SeqCollator):
    """Drops all the suffix/target tokens and sends them in the target_ids of shape (batch_size, target_seq_len)"""

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> ILMBatch:
        # handle the prefix
        cls_token_id = self.tokenizer.cls_token_id
        prefix = prepare_prefix_ids(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            max_seq_len=self.input_block_size,
            cls_token_id=cls_token_id,
            bos_token_id=self.tokenizer.bos_token_id
        )
        target_ids = prepare_target_ids_for_test(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            max_seq_len=self.block_size,
            bos_token_id=None,  # BOS will be preped by the prefix
        )
        return {
            "input_ids": prefix["input_ids"],
            "attention_mask": prefix["attention_mask"],
            "token_type_ids": prefix["token_type_ids"],
            "target_ids": target_ids["target_ids"],
            "target_attention_mask": target_ids["attention_mask"],
            "n_drops": None,
            "constraint": None,
            "cls_position": prefix["cls_position"],
        }


# endregion: Collators
################################################################################


################################################################################
# region: Utilities


def print_batch_ilm(
    batch: Dict[str, Any],
    split: Literal["train", "val", "test", "predict"],
    tokenizer: Tokenizer,
    dataloader_name: str = "",
):

    logger.info(
        f"Printing first entries of the tensors in batch for {split}/{dataloader_name}..."
    )
    print("input tokens:")
    print(tokenizer.decode(batch["input_ids"][0]))
    print("input_ids:")
    print(batch["input_ids"][0])
    print("attention_mask (int):")
    print(batch["attention_mask"][0].int())
    print("token_type_ids:")
    print(batch["token_type_ids"][0])
    if batch.get("n_drops", None) is not None:
        print("n_drops:")
        print(batch["n_drops"][0])
    if batch.get("target_attention_mask", None) is not None:
        print("target_attention_mask:")
        print(batch["target_attention_mask"][0])
    print("target_ids:")
    print(
        batch["target_ids"][0].to_sparse()
        if batch is not None and batch.get("target_ids", None) is not None
        else None
    )
    print("constraint:")
    print(
        batch["constraint"][0]
        if batch.get("constraint", None) is not None
        else None
    )
    print("cls_position:")
    print(
        batch["cls_position"][0] if batch["cls_position"] is not None else None
    )


# endregion: Utilities
################################################################################


################################################################################
# region: Prediction utils


# endregion: Prediction utils
################################################################################
