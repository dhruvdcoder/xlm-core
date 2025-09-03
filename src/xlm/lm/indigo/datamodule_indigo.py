from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Any,
    Optional,
    Tuple,
)
from jaxtyping import Integer
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
from xlm.utils.nn import pad_truncate_list
from .types_indigo import IndigoBatch

logger = RankedLogger(__name__, rank_zero_only=True)

################################################################################
# region: Dataset


class IndigoEmptyDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        num_examples: int,
        min_len: int,
        max_len: int,
        uniform: bool = False,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.min_len = min_len
        self.max_len = max_len
        self.uniform = (
            uniform  # true if yield all examples w same fixed length
        )
        # else length sampled uniformaly random in [min_len, max_len]
        self.rng = np.random.default_rng(seed)

        # Collect special token ids that does not appear in content
        special_ids = {
            tid
            for attr in [
                "pad_token_id",
                "bos_token_id",
                "eos_token_id",
                "unk_token_id",
                "cls_token_id",
                "sep_token_id",
                "mask_token_id",
                "eod_token_id",
            ]
            if (tid := getattr(tokenizer, attr, None)) is not None
        }
        if hasattr(tokenizer, "additional_special_tokens_ids"):
            special_ids.update(tokenizer.additional_special_tokens_ids)

        self.normal_ids = np.array(
            [i for i in range(len(tokenizer)) if i not in special_ids],
            dtype=np.int64,
        )
        if len(self.normal_ids) == 0:
            raise ValueError("No non-special tokens available to sample.")

    def __iter__(self):
        fixed_len = (self.max_len + self.min_len) // 2
        for _ in range(self.num_examples):
            L = (
                fixed_len
                if self.uniform
                else int(self.rng.integers(self.min_len, self.max_len + 1))
            )
            ids = self.rng.choice(
                self.normal_ids, size=L, replace=True
            ).tolist()
            yield {"target_ids": ids}


# endregion: Dataset
################################################################################


def prepare_single_suffix_sequence_indigo(
    single_suffix_ids_sequence: List[int],
    eos_token_id: int,
    eod_token_id: int,
    bos_token_id: Optional[int],
    global_offset: int = 0,
):
    add_bos = int(bos_token_id is not None)
    if add_bos:
        _pi = (
            global_offset + 2 + torch.randperm(len(single_suffix_ids_sequence))
        ).tolist()
        pi = (
            [global_offset + 0, global_offset + 1]
            + _pi
            + [len(single_suffix_ids_sequence) + global_offset + 2]
        )  # bos, eos, permuted_suffix, eod
        _suffix_ids = (
            [bos_token_id, eos_token_id]
            + single_suffix_ids_sequence
            + [eod_token_id]
        )
    else:
        _pi = (
            global_offset + 1 + torch.randperm(len(single_suffix_ids_sequence))
        ).tolist()
        pi = (
            [global_offset + 0]
            + _pi
            + [len(single_suffix_ids_sequence) + global_offset + 1]
        )
        _suffix_ids = (
            [eos_token_id] + single_suffix_ids_sequence + [eod_token_id]
        )
    return _suffix_ids, pi


def permute_suffix_ids_indigo(
    suffix_ids: List[List[int]],
    eos_token_id: int,
    eod_token_id: int,
    bos_token_id: Optional[int] = None,
    global_offsets: Optional[List[int]] = None,
    max_seq_len: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """

    Args:
        suffix_ids: List of suffix token sequences.
        eos_token_id: EOS token ID.
        eod_token_id: End of decoding token ID.
        bos_token_id: BOS token ID.
        global_offset: Global offset for the permutation.
        max_seq_len: Maximum sequence length.

    Returns:
        input_ids: List of input ids.
        attention_mask: List of attention masks.
        pi: List of permutation indices.
    """
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    pi: List[List[int]] = []
    if global_offsets is None:
        global_offsets = [0] * len(suffix_ids)

    # Determine max length
    max_suffix_len = max(len(_suffix_ids) for _suffix_ids in suffix_ids)
    if max_seq_len is not None:
        if max_suffix_len + 3 > max_seq_len:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) must be greater than max_suffix_len ({max_suffix_len}) + 3"
            )
    else:
        max_seq_len = max_suffix_len + 3

    for i, single_suffix_ids_sequence in enumerate(suffix_ids):
        single_suffix, _pi = prepare_single_suffix_sequence_indigo(
            single_suffix_ids_sequence,
            eos_token_id,
            eod_token_id,
            bos_token_id,
            global_offsets[i],
        )
        _attention_mask = [1] * len(single_suffix)
        # we don't pad here, pad in the collator
        # extend pi to max_seq_len

        pi.append(_pi)
        input_ids.append(single_suffix)
        attention_mask.append(_attention_mask)
    return input_ids, attention_mask, pi


# subject to change
def build_insertion_trajectory_random(
    target_tokens: List[int],
    bos_id: int,
    eos_id: int,
    eod_id: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, TT]:
    """
    Create a random permutation order and simulate insertion trajectory.

    Returns:
        word_labels: Long[T+1]
        pointer_labels: Long[T+1] (last is -100 for <eod>)
        relative_matrix: Int8[(T+2),(T+2)]
        absolute_positions: Long[T+2]
        order_indices: Long[T] (the random permutation itself)
    """
    if rng is None:
        rng = np.random.default_rng()
    T = len(target_tokens)
    if T == 0:
        # only predict <eod>
        R = torch.tensor([[0, 1], [-1, 0]], dtype=torch.int8)
        return {
            "word_labels": torch.tensor([eod_id], dtype=torch.long),
            "pointer_labels": torch.tensor([-100], dtype=torch.long),
            "relative_matrix": R,
            "absolute_positions": torch.tensor([-1, 0], dtype=torch.long),
            "order_indices": torch.tensor([], dtype=torch.long),
        }

    order_perm = rng.permutation(T).tolist()

    # Current sequence
    current: List[Tuple[int, int]] = [(-1, bos_id), (T, eos_id)]
    R = torch.tensor(
        [[0, 1], [-1, 0]], dtype=torch.int8
    )  # initial relative matrix

    word_labels: List[int] = []
    pointer_labels: List[int] = []

    token_by_abs = {i: tok for i, tok in enumerate(target_tokens)}

    def rebuild_sorted():
        current.sort(key=lambda x: x[0])

    def pointer_slot(anchor_pos: int, side: int) -> int:
        return 2 * anchor_pos if side == -1 else 2 * anchor_pos + 1

    for abs_index in order_perm:
        token_id = token_by_abs[abs_index]
        rebuild_sorted()
        abs_list = [a for a, _ in current]

        greater = [a for a in abs_list if a > abs_index]
        if greater:
            anchor_abs = min(greater)
            side = -1
        else:
            anchor_abs = max(abs_list)
            side = 1

        anchor_pos = abs_list.index(anchor_abs)
        slot = pointer_slot(anchor_pos, side)

        # Construct new row
        anchor_row = R[anchor_pos].clone()
        anchor_row[anchor_pos] = -1 if side == -1 else 1

        # Expand matrix
        R = torch.cat([R, anchor_row.unsqueeze(0)], dim=0)
        new_col = torch.cat([-anchor_row, torch.tensor([0], dtype=torch.int8)])
        R = torch.cat([R, new_col.unsqueeze(1)], dim=1)

        current.append((abs_index, token_id))
        word_labels.append(token_id)
        pointer_labels.append(slot)

    # Final <eod> step
    word_labels.append(eod_id)
    pointer_labels.append(-100)  # ignore in loss

    rebuild_sorted()
    absolute_positions = torch.tensor(
        [a for a, _ in current], dtype=torch.long
    )

    return {
        "word_labels": torch.tensor(word_labels, dtype=torch.long),
        "pointer_labels": torch.tensor(pointer_labels, dtype=torch.long),
        "relative_matrix": R,
        "absolute_positions": absolute_positions,
        "order_indices": torch.tensor(order_perm, dtype=torch.long),
    }


################################################################################
# region: Collators


class DefaultIndigoCollator(Collator):
    """Used for pre-training."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        truncate: Literal["max", "block", None] = "block",
        seed: Optional[int] = None,  # for randomness
    ):
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self._vocab_size = len(self.tokenizer)
        self.truncate = truncate
        self._vocab_size = len(tokenizer)
        self._bos = tokenizer.bos_token_id
        self._eos = tokenizer.eos_token_id
        self._pad = tokenizer.pad_token_id
        self._eod = getattr(tokenizer, "eod_token_id", self._eos)
        self.rng = np.random.default_rng(seed)

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not set")
            self._vocab_size = len(self.tokenizer)
        return self._vocab_size

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def _truncate(self, seq: List[int]) -> List[int]:
        if self.truncate == "block" and len(seq) > self.block_size:
            return seq[: self.block_size]
        return seq

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> IndigoBatch:
        # TODO (URV): Implement the collator.
        if not examples:
            raise ValueError("Empty examples passed to collator.")

        targets = [
            self._truncate(
                list(e.get("target_ids") or e.get("input_ids") or [])
            )
            for e in examples
        ]

        batch_size = len(targets)
        max_tgt_len = max(len(t) for t in targets) if targets else 1
        pad_id = self._pad

        target_ids = torch.full(
            (batch_size, max_tgt_len), pad_id, dtype=torch.long
        )
        target_attention_mask = torch.zeros_like(target_ids, dtype=torch.bool)

        word_label_list: List[TT] = []
        pointer_label_list: List[TT] = []
        order_list: List[TT] = []
        rel_mats: List[TT] = []
        abs_pos_list: List[TT] = []
        traj_lengths: List[int] = []
        tgt_lengths: List[int] = []

        for i, tgt in enumerate(targets):
            L = len(tgt)
            tgt_lengths.append(L)
            target_ids[i, :L] = torch.tensor(tgt, dtype=torch.long)
            target_attention_mask[i, :L] = True

            traj = build_insertion_trajectory_random(
                target_tokens=tgt,
                bos_id=self._bos,
                eos_id=self._eos,
                eod_id=self._eod,
                rng=self.rng,
            )
            word_label_list.append(traj["word_labels"])
            pointer_label_list.append(traj["pointer_labels"])
            order_list.append(traj["order_indices"])
            rel_mats.append(traj["relative_matrix"])
            abs_pos_list.append(traj["absolute_positions"])
            traj_lengths.append(traj["word_labels"].size(0))

        # Pad sequences
        max_steps = (
            max(w.size(0) for w in word_label_list) if word_label_list else 1
        )
        word_labels = torch.full(
            (batch_size, max_steps), pad_id, dtype=torch.long
        )
        word_labels_mask = torch.zeros(
            (batch_size, max_steps), dtype=torch.bool
        )
        pointer_labels = torch.full(
            (batch_size, max_steps), -100, dtype=torch.long
        )
        pointer_labels_mask = torch.zeros(
            (batch_size, max_steps), dtype=torch.bool
        )

        for i, (wl, pl) in enumerate(zip(word_label_list, pointer_label_list)):
            s = wl.size(0)
            word_labels[i, :s] = wl
            word_labels_mask[i, :s] = True
            pointer_labels[i, :s] = pl
            pointer_labels_mask[i, :s] = pl.ne(-100)

        # Pad order permutations
        max_order = max(o.size(0) for o in order_list) if order_list else 0
        order_indices = torch.full(
            (batch_size, max_order), -100, dtype=torch.long
        )
        for i, oi in enumerate(order_list):
            order_indices[i, : oi.size(0)] = oi

        # Pad relative matrices
        rel_size = max_tgt_len + 2
        relative_matrix = torch.zeros(
            (batch_size, rel_size, rel_size), dtype=torch.int8
        )
        absolute_positions = torch.zeros(
            (batch_size, rel_size), dtype=torch.long
        )
        for i, R in enumerate(rel_mats):
            sz = R.size(0)
            relative_matrix[i, :sz, :sz] = R
        for i, ap in enumerate(abs_pos_list):
            absolute_positions[i, : ap.size(0)] = ap

        # Dummy encoder (no source for decoder-only pretraining)
        input_ids = torch.zeros((batch_size, 1), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        batch: IndigoBatch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target_attention_mask": target_attention_mask,
            "order_indices": order_indices,
            "word_labels": word_labels,
            "word_labels_mask": word_labels_mask,
            "pointer_labels": pointer_labels,
            "pointer_labels_mask": pointer_labels_mask,
            "relative_matrix": relative_matrix,
            "step_relative_vectors": None,  # Not built here
            "absolute_positions": absolute_positions,
            "target_lengths": torch.tensor(tgt_lengths, dtype=torch.long),
            "trajectory_lengths": torch.tensor(traj_lengths, dtype=torch.long),
            "order_name": ["RND"] * batch_size,
        }
        return batch


class IndigoSeq2SeqCollator:
    """Used for seq2seq training.
    Prepares joint sequences that are only padded on the right.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        noise_schedule: NoiseSchedule,
        block_size: int,
        input_block_size: int,
        bos_before_prefix: bool = False,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.input_block_size = input_block_size
        self._vocab_size = (
            len(self.tokenizer) if self.tokenizer is not None else None
        )
        self._bos = tokenizer.bos_token_id
        self._eos = tokenizer.eos_token_id
        self._pad = tokenizer.pad_token_id
        self._eod = tokenizer.cls_token_id
        self.bos_before_prefix = bos_before_prefix
        # we will not support truncation for now
        if self.bos_before_prefix:
            raise NotImplementedError(
                "TODO: Add support for having BOS before prefix."
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
    ) -> IndigoBatch:

        # prefix/source: we only check lengths and create increasing permutation lists
        global_offsets: List[int] = []
        prefix_ids: List[List[int]] = []
        prefix_attention_masks: List[List[int]] = []
        prefix_pis: List[List[int]] = []
        for ex in examples:
            _prefix_ids = ex["input_ids"]
            if len(_prefix_ids) > self.input_block_size:
                raise ValueError(
                    f"Prefix length {len(_prefix_ids)} exceeds input block size {self.input_block_size}"
                )
            global_offsets.append(len(_prefix_ids))
            prefix_ids.append(_prefix_ids)
            prefix_attention_masks.append([1] * len(_prefix_ids))
            prefix_pis.append(list(range(len(_prefix_ids))))

        suffix_ids, suffix_attention_masks, suffix_pis = (
            permute_suffix_ids_indigo(
                suffix_ids=examples["input_ids"],
                eos_token_id=self._eos,
                eod_token_id=self._eod,
                bos_token_id=self._bos,
                global_offsets=global_offsets,
            )
        )
        if len(prefix_ids) != len(suffix_ids):
            raise RuntimeError(
                f"Prefix and suffix lengths do not match: {len(prefix_ids)} != {len(suffix_ids)}"
            )
        # pad after joining prefix and suffix
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        pi: List[List[int]] = []
        target_ids: List[List[int]] = []

        max_joint_len = self.input_block_size + self.block_size
        for i in range(len(prefix_ids)):
            input_ids.append(
                pad_truncate_list(
                    prefix_ids[i] + suffix_ids[i],
                    max_joint_len,
                    pad_token_id=self._pad,
                    pad_left=False,
                )
            )
            attention_mask.append(
                pad_truncate_list(
                    prefix_attention_masks[i] + suffix_attention_masks[i],
                    max_joint_len,
                    pad_token_id=0,
                    pad_left=False,
                )
            )
            pi.append(
                prefix_pis[i]
                + suffix_pis[i]
                + list(
                    range(
                        len(prefix_ids[i]) + len(suffix_ids[i]), max_joint_len
                    )
                )
            )
            # TODO (bos_before_prefix): last token of prefix acts like BOS when bos_before_prefix is True
            target_ids.append(
                pad_truncate_list(
                    [-100]
                    * len(
                        prefix_ids[i] + 1
                    )  # prefix + BOS have no target to predict
                    + suffix_ids[i][
                        2:
                    ],  # EOS onwards we have targets that are simply the suffix tokens but shifted by 1
                    max_joint_len,
                    pad_token_id=self._pad,
                    pad_left=False,
                )
            )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "pi": torch.tensor(pi, dtype=torch.long),
        }


class IndigoSeq2SeqPredCollator(IndigoSeq2SeqCollator):
    """Drops all the suffix/target tokens and sends them in the target_ids of shape (batch_size, target_seq_len)"""

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> Dict[str, Any]:
        # TODO (URV): Implement the collator.
        if not examples:
            raise ValueError("Empty examples in prediction collator.")

        sources = [
            self._truncate(list(ex["input_ids"]), self.input_block_size)
            for ex in examples
        ]
        targets = [
            self._truncate(list(ex["labels"]), self.block_size)
            for ex in examples
        ]

        pad_id = self._pad
        batch_size = len(sources)

        # Source
        max_src = max(len(s) for s in sources) if sources else 1
        input_ids = torch.full((batch_size, max_src), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, s in enumerate(sources):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, : len(s)] = True

        # Target
        max_tgt = max(len(t) for t in targets) if targets else 1
        target_ids = torch.full(
            (batch_size, max_tgt), pad_id, dtype=torch.long
        )
        target_attention_mask = torch.zeros_like(target_ids, dtype=torch.bool)
        for i, t in enumerate(targets):
            target_ids[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            target_attention_mask[i, : len(t)] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target_attention_mask": target_attention_mask,
            # Trajectory fields are None
            "order_indices": None,
            "word_labels": None,
            "word_labels_mask": None,
            "pointer_labels": None,
            "pointer_labels_mask": None,
            "relative_matrix": None,
            "absolute_positions": None,
            "step_relative_vectors": None,
            "trajectory_lengths": None,
            "target_lengths": torch.tensor(
                [len(t) for t in targets], dtype=torch.long
            ),
            "order_name": None,
        }


# endregion: Collators
################################################################################


################################################################################
# region: Utilities


def print_batch_indigo(
    batch: Dict[str, Any],
    split: Literal["train", "val", "test", "predict"],
    tokenizer: Tokenizer,
    dataloader_name: str = "",
    max_examples: int = 1,
):
    # TODO (URV)
    logger.info(f"[Indigo] split={split} dl={dataloader_name}")
    if "target_ids" not in batch or batch["target_ids"] is None:
        print("No target_ids present.")
        return

    show = min(max_examples, batch["target_ids"].size(0))
    for i in range(show):
        print(f"--- Example {i} ---")
        if batch.get("input_ids") is not None:
            src_ids = batch["input_ids"][i]
            src_mask = batch.get(
                "attention_mask", torch.ones_like(src_ids)
            ).bool()
            src_tokens = [
                tokenizer.decode([int(t)]) for t in src_ids[src_mask]
            ]
            print("Source:", src_tokens)

        tgt_ids = batch["target_ids"][i]
        tgt_mask = batch.get("target_attention_mask")
        if tgt_mask is not None:
            tgt_mask = tgt_mask[i].bool()
        else:
            tgt_mask = tgt_ids.ne(tokenizer.pad_token_id)
        tgt_tokens = [tokenizer.decode([int(t)]) for t in tgt_ids[tgt_mask]]
        print("Target:", tgt_tokens)

        if batch.get("word_labels") is not None:
            wl = batch["word_labels"][i]
            wl_mask = batch["word_labels_mask"][i].bool()
            wl_tokens = [
                tokenizer.decode([int(t)])
                for t in wl[wl_mask]
                if int(t) != tokenizer.pad_token_id
            ]
            print("Word labels (gen order):", wl_tokens)

        if batch.get("pointer_labels") is not None:
            pl = batch["pointer_labels"][i]
            pl_mask = batch["pointer_labels_mask"][i].bool()
            print("Pointer labels:", pl[pl_mask].tolist())

        if batch.get("order_indices") is not None:
            oi = batch["order_indices"][i]
            oi = oi[oi.ge(0)]
            print("Order permutation:", oi.tolist())

        if batch.get("relative_matrix") is not None:
            R = batch["relative_matrix"][i]
            print("Relative matrix shape:", tuple(R.shape))
        print("--------------------")


# endregion: Utilities
################################################################################
