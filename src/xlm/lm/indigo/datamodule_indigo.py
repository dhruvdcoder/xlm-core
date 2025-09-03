from typing import (
    Dict,
    List,
    Literal,
    Any,
    Optional,
    Tuple,
)
from torch.utils.data import IterableDataset
import torch
from xlm.noise import NoiseSchedule
from xlm.utils.rank_zero import RankedLogger
from xlm.datamodule import (
    Seq2SeqCollatorInput,
    Tokenizer,
    Collator,
    BaseCollatorInput,
)
from xlm.utils.nn import pad_truncate_list
from .types_indigo import (
    IndigoBatch,
    IndigoSeq2SeqPredBatch,
    IndigoSeq2SeqBatch,
)

logger = RankedLogger(__name__, rank_zero_only=True)

################################################################################
# region: Dataset


class IndigoEmptyDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        num_examples: int,
    ):
        """Initialize the Indigo empty dataset.

        Args:
            tokenizer: The tokenizer to use.
            num_examples: Number of empty examples to generate.
        """
        self.tokenizer = tokenizer
        self.num_examples = num_examples

    def __iter__(self):
        """Generate empty examples for Indigo training."""
        for _ in range(self.num_examples):
            ex = self.tokenizer(
                "",
                add_special_tokens=False,
            )
            yield ex


# endregion: Dataset
################################################################################


def get_suffix_permutation_indigo(
    single_suffix_ids_sequence: List[int],
    eos_token_id: int,
    eod_token_id: int,
    bos_token_id: Optional[int],
    global_offset: int = 0,
) -> Tuple[List[int], List[int]]:
    """
    Prepare a single suffix sequence for Indigo.

    Returns:
        suffix_ids: List of suffix token sequences with global offset, BOS(optional), EOS, seq, EOD
        pi: List of permutation indices.
    """
    add_bos = int(bos_token_id is not None)
    # .... BOS EOS t1 .... tn EOD PAD PAD
    # .... 0   n+1 r1 .... rn n+2 n+3 n+4,    r1 starts from 1
    if add_bos:
        perm = torch.randperm(len(single_suffix_ids_sequence))
        _pi = (global_offset + 1 + perm).tolist()
        pi = (
            [
                global_offset + 0,
                global_offset + len(single_suffix_ids_sequence) + 1,
            ]
            + _pi
            + [len(single_suffix_ids_sequence) + global_offset + 2]
        )  # bos, eos, permuted_suffix, eod
        permuted_suffix_ids = (
            [bos_token_id, eos_token_id]
            + [single_suffix_ids_sequence[i] for i in perm]
            + [eod_token_id]
        )
    else:
        # .... EOS t1 .... tn EOD PAD PAD
        # .... n   r1 .... rn n+1,      r1 starts from 0
        perm = torch.randperm(len(single_suffix_ids_sequence))
        _pi = (global_offset + 0 + perm).tolist()
        pi = (
            [global_offset + 0]
            + _pi
            + [len(single_suffix_ids_sequence) + global_offset + 1]
        )
        permuted_suffix_ids = (
            [eos_token_id]
            + [single_suffix_ids_sequence[i] for i in perm]
            + [eod_token_id]
        )
    return permuted_suffix_ids, pi


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
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.truncate = truncate
        self._vocab_size = len(tokenizer)
        if getattr(tokenizer, "cls_token_id", None) is None:
            raise ValueError("Tokenizer must have a cls_token_id")

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not set")
            self._vocab_size = len(self.tokenizer)
        return self._vocab_size

    def _truncate(self, seq: List[int]) -> List[int]:
        if self.truncate == "block" and len(seq) > self.block_size:
            return seq[: self.block_size]
        return seq

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> IndigoBatch:
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        target_ids: List[List[int]] = []
        pis: List[List[int]] = []
        # extract input_ids from examples
        seq_lens = [len(e["input_ids"]) for e in examples]

        # determine max length based on truncation strategy
        tokens_to_add = 3  # EOS, BOS, EOD
        if self.truncate == "max":
            max_len = min(max(seq_lens) + tokens_to_add, self.block_size)
        elif self.truncate == "block":
            max_len = self.block_size
        elif self.truncate is None:
            max_len = max(seq_lens) + tokens_to_add
        else:
            raise ValueError(f"Invalid truncate value: {self.truncate}")

        for example in examples:
            # get the input sequence
            seq = example["input_ids"]

            # truncate if necessary (account for BOS, EOS, EOD tokens)
            if len(seq) > max_len - tokens_to_add:
                seq = seq[: max_len - tokens_to_add]

            # generate the permutation
            # BOS EOS t1 ... tn EOD
            permuted_seq, pi = get_suffix_permutation_indigo(
                seq,
                eos_token_id=self.tokenizer.eos_token_id,
                eod_token_id=self.tokenizer.cls_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                global_offset=0,
            )
            # REMOVE THIS. Only left for reference.
            # pi = (
            #    [0, len(seq) + 1]
            #    + (1 + np.random.permutation(len(seq))).tolist()
            #    + [len(seq) + 2]
            # )
            # permuted_seq = (
            #    [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
            #    + [seq[i] for i in pi]
            #    + [self.tokenizer.cls_token_id]
            # )
            tgt_ids = permuted_seq[1:] + [-100]
            # 1 for all non-pad tokens
            attn_mask = [1] * len(permuted_seq)

            # pad to max_len
            padded_seq = pad_truncate_list(
                permuted_seq,
                max_len,
                self.tokenizer.pad_token_id,
                pad_left=False,
            )
            attn_mask = pad_truncate_list(
                attn_mask,
                max_len,
                0,
                pad_left=False,
            )
            tgt_ids = pad_truncate_list(
                tgt_ids,
                max_len,
                -100,
                pad_left=False,
            )
            # for pi, we simply increment for PAD tokens, they will be ignored in the loss anyway
            pi = pi + list(range(len(pi) + 1, len(target_ids) + 1))
            assert len(set(pi)) == len(pi), "Duplicate entries in pi"
            assert len(pi) == len(
                tgt_ids
            ), "pi and target_ids must have the same length"
            assert (
                max(pi) == len(tgt_ids) - 1
            ), "pi must be a permutation of the target_ids"

            input_ids.append(padded_seq)
            attention_mask.append(attn_mask)
            target_ids.append(tgt_ids)
            pis.append(pi)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "pi": torch.tensor(pis, dtype=torch.long),
        }


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
    ) -> IndigoSeq2SeqBatch:
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        pis: List[List[int]] = []
        target_ids: List[List[int]] = []

        for ex in examples:
            _prefix_ids = ex["prompt_ids"]
            if len(_prefix_ids) > self.input_block_size:
                raise ValueError(
                    f"Prefix length {len(_prefix_ids)} exceeds input block size {self.input_block_size}"
                )
            _suffix_ids = ex["input_ids"]
            if len(_suffix_ids) > self.block_size:
                raise ValueError(
                    f"Suffix length {len(_suffix_ids)} exceeds block size {self.block_size}"
                )
            permuted_suffix_ids, pi = get_suffix_permutation_indigo(
                _suffix_ids,
                eos_token_id=self._eos,
                eod_token_id=self._eod,
                bos_token_id=None,
                global_offset=len(_prefix_ids),
            )
            # joint sequence
            ids = _prefix_ids + permuted_suffix_ids
            # we don't want to predict the prompt tokens
            tgt_ids = [-100] * len(_prefix_ids) + permuted_suffix_ids
            attn_mask = pad_truncate_list(
                [1] * len(ids),
                len(_prefix_ids) + self.block_size,
                0,
                pad_left=False,
            )
            pi = list(range(len(_prefix_ids))) + pi
            tgt_ids = pad_truncate_list(
                tgt_ids[1:] + [-100],
                len(_prefix_ids) + self.block_size,
                -100,
                pad_left=False,
            )
            input_ids.append(ids)
            attention_mask.append(attn_mask)
            target_ids.append(tgt_ids)
            pis.append(pi)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "pi": torch.tensor(pis, dtype=torch.long),
        }


class IndigoSeq2SeqPredCollator(IndigoSeq2SeqCollator):
    """Drops all the suffix/target tokens and sends them in the target_ids of shape (batch_size, prefix_len + target_seq_len), which is different from ARLM which only sends the target_ids of shape (batch_size, target_seq_len). Also pads prompt on the left so all input sequences end at the same position."""

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> IndigoSeq2SeqPredBatch:
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        target_ids: List[List[int]] = []
        pis: List[List[int]] = []
        for ex in examples:
            _prefix_ids = ex["prompt_ids"]
            if len(_prefix_ids) > self.input_block_size:
                raise ValueError(
                    f"Prefix length {len(_prefix_ids)} exceeds input block size {self.input_block_size}"
                )
            # we need to send in BOS EOS as part of the prefix
            prefix_ids = _prefix_ids + [self._bos, self._eos]
            attn_mask = pad_truncate_list(
                [1] * len(prefix_ids),
                self.input_block_size,
                0,
                pad_left=True,
            )
            pi = list(range(len(_prefix_ids)))
            pis.append(pi)
            input_ids.append(_prefix_ids)
            attention_mask.append(attn_mask)
            _suffix_ids = ex["input_ids"]
            if len(_suffix_ids) > self.block_size:
                raise ValueError(
                    f"Suffix length {len(_suffix_ids)} exceeds block size {self.block_size}"
                )
            tgt_ids = prefix_ids + [self._bos] + _suffix_ids + [self._eos]
            target_ids.append(tgt_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "pi": torch.tensor(pis, dtype=torch.long),
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
):
    logger.info(f"[Indigo] split={split} dl={dataloader_name}")
    print("input tokens:")
    print(tokenizer.decode(batch["input_ids"][0]))
    print("input_ids:")
    print(batch["input_ids"][0])
    print("attention_mask (int):")
    print(batch["attention_mask"][0].int())
    print("pi:")
    print(batch["pi"][0])
    if batch.get("target_ids") is not None:
        print("target_ids:")
        print(batch["target_ids"][0])


# endregion: Utilities
################################################################################
