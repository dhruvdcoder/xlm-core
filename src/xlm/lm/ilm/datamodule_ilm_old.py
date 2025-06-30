import collections
from typing import (
    Dict,
    List,
    Literal,
    Any,
    Optional,
)
from jaxtyping import Integer, Bool
from torch import Tensor as TT
from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import collate_tensor_fn
import numpy as np
import torch
from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    GPT2TokenizerFast,
    PreTrainedTokenizerFast,
)
from xlm.noise import NoiseSchedule
from xlm.utils.rank_zero import RankedLogger
from tokenizers import processors
from xlm.datamodule import (
    Tokenizer,
    BaseDataModule,
    Collator,
    BaseCollatorInput,
)
from .types_ilm import ILMBatch
import datasets

logger = RankedLogger(__name__, rank_zero_only=True)


################################################################################
# region: Tokenizers


class ILMTokenizerMixin:
    """Overrides the two key methods.
    Should be used as a mixin with mro order:
    class IDLMTokenizer(IDLMTokenizerMixin, PreTrainedTokenizerFast):
        pass
    Note:
      Make sure to call `post_creation` after initializing the tokenizer.
    """

    @property
    def full_vocab_size(self) -> int:
        return self.__len__()

    def post_creation(self):
        """Check the presence of the special tokens and update the post processor."""
        for special_token in [
            "eos_token",
            "bos_token",
            "cls_token",
            "pad_token",
            "mask_token",
            "sep_token",
            "unk_token",
        ]:
            if (token := getattr(self, special_token)) is None:
                raise ValueError(f"{special_token} is not set")
        if isinstance(self, PreTrainedTokenizerFast):
            self.update_post_processor()

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        # token_ids_1 is assumed to be a prefix
        if token_ids_1 is not None:
            return (
                [self.cls_token_id]  # type: ignore
                + [self.bos_token_id]  # type: ignore
                + token_ids_1
                + token_ids_0
            )  # type: ignore
        else:
            return [self.cls_token_id] + [self.bos_token_id] + token_ids_0  # type: ignore

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        # token_ids_1 is assumed to be a prefix
        if token_ids_1 is None:
            return [0, 1] + [2] * len(token_ids_0)  # type: ignore
        else:
            return [0, 1] + [1] * len(token_ids_1) + [2] * len(token_ids_0)  # type: ignore

    def create_post_processor(self) -> processors.TemplateProcessing:
        if (self.cls_token is None) or (self.cls_token_id is None):
            raise ValueError("cls_token is required.")
        if (self.bos_token is None) or (self.bos_token_id is None):
            raise ValueError("bos_token is required.")
        post_processor = processors.TemplateProcessing(
            single=f"{self.cls_token}:0 {self.bos_token}:1 $A:2",
            pair=f"{self.cls_token}:0 {self.bos_token}:1 $B:1 $A:2",
            special_tokens=[
                (self.cls_token, self.cls_token_id),
                (self.bos_token, self.bos_token_id),
            ],
        )
        return post_processor

    def update_post_processor(self) -> None:
        self.post_processor = self.create_post_processor()
        self._tokenizer.post_processor = self.post_processor


# The specific tokenizers can go in the dataset specific files.
class BertTokenizerForILM(ILMTokenizerMixin, BertTokenizer):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens({"cls_token": "[CLS]", "bos_token": "[BOS]"})

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


class BertTokenizerForILMFast(ILMTokenizerMixin, BertTokenizerFast):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens(
            {"cls_token": "[CLS]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


class GPT2TokenizerForILMFast(ILMTokenizerMixin, GPT2TokenizerFast):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens(
            {
                "cls_token": "<|cls|>",
                "bos_token": "<|bos|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "mask_token": "<|mask|>",
                "sep_token": "<|sep|>",
                "eos_token": "<|endoftext|>",  # original
            }
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


# endregion: Tokenizers
################################################################################


################################################################################
# region: Processors


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
                add_special_tokens=True,
            )
            yield ex


class ILMUnconditionalGenerationDatasetManager:

    def __init__(
        self,
        num_examples: int,
        split_by_node: bool = True,
    ):
        self.num_examples = num_examples
        self.split_by_node = split_by_node

    def __repr__(self) -> str:
        return f"ILMUnconditionalGenerationDatasetManager()"

    @property
    def name(self) -> str:
        return "empty"

    def prepare_data(
        self,
        manual_cache_dir: str,
        tokenizer: Tokenizer,
        num_proc: Optional[int] = None,
    ) -> None:
        return

    def setup(
        self,
        stage: Literal["fit", "validate", "test", "predict"],
        manual_cache_dir: str,
        tokenizer: Tokenizer,
        block_size: int,
        is_ddp: bool,
        rank: int,
        world_size: int,
    ) -> None:
        if is_ddp and world_size > 1 and self.split_by_node:
            examples_per_node = self.num_examples // world_size
        else:
            examples_per_node = self.num_examples
        dataset = ILMEmptyDataset(
            tokenizer=tokenizer,
            num_examples=examples_per_node,
        )
        self.dataset = dataset


# endregion: Processors
################################################################################


################################################################################
# region: Collators


class DefaultILMCollator(Collator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        loss_on_padding: bool = True,
        return_dense_target: bool = False,  # setting to two will increase the cpu memory usage
    ):
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self.loss_on_padding = loss_on_padding
        self.return_dense_target = return_dense_target
        if self.loss_on_padding:
            self.attn_extension_id = 1
        else:
            self.attn_extension_id = 0

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return len(self.tokenizer)

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def sample_n_drops(self, seq_len: int) -> int:
        return np.random.randint(seq_len + 1)

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> ILMBatch:
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        drop_masks: List[Bool[TT, " max_seq_len"]] = []
        # other tensors
        input_ids: List[Integer[TT, " max_seq_len"]] = []
        attention_mask: List[Integer[TT, " max_seq_len"]] = []
        token_type_ids: List[Integer[TT, " max_seq_len"]] = []
        # Elements of the sparse tensor
        batch_indices: List[int] = []
        seq_indices: List[int] = []
        vocab_indices: List[int] = []
        values: List[int] = []
        for e, _example in enumerate(examples):
            if len(_example["input_ids"]) > max_seq_len:
                raise ValueError(
                    f"Input ids length {len(_example['input_ids'])} is greater than max_seq_len {max_seq_len}"
                )
            example = {
                "input_ids": torch.tensor(
                    _example["input_ids"][:max_seq_len]
                    + (max_seq_len - len(_example["input_ids"]))
                    * [self.tokenizer.pad_token_id]
                ),
                "attention_mask": torch.tensor(
                    _example["attention_mask"][:max_seq_len]
                    + (max_seq_len - len(_example["attention_mask"]))
                    * [self.attn_extension_id]
                ).bool(),
                "token_type_ids": torch.tensor(
                    _example["token_type_ids"][:max_seq_len]
                    + (max_seq_len - len(_example["token_type_ids"])) * [2]
                ),
            }
            non_pad = example["attention_mask"]
            input_ids.append(example["input_ids"])
            attention_mask.append(non_pad)
            token_type_ids.append(example["token_type_ids"])
            non_prefix = (
                example["token_type_ids"] > 1
                if "token_type_ids" in example
                else torch.ones_like(non_pad)
            )
            non_pad_and_non_prefix_indices = torch.logical_and(
                non_pad, non_prefix
            ).nonzero(as_tuple=True)[
                0
            ]  # shape: (*n_non_pad_and_non_prefix,)
            _seq_len = int(len(non_pad_and_non_prefix_indices))
            _n_drops = self.sample_n_drops(_seq_len)
            _pre_drop_indices = (torch.randperm(_seq_len))[
                :_n_drops
            ]  # shape: (n_drops,)
            _drop_indices = non_pad_and_non_prefix_indices[
                _pre_drop_indices
            ]  # shape: (n_drops,)
            _drop_mask = torch.zeros((max_seq_len,), dtype=torch.bool)
            _drop_mask[_drop_indices] = True
            drop_masks.append(_drop_mask)
            # we maintain two indices:
            # 1. The index of the last non-dropped token: prev_remaining_j,
            #   initialized to the index of the token right before the first dropped token
            # 2. The current index: j
            # Check for empty tensor to handle the case blank input during prediction
            start = (
                int(non_pad_and_non_prefix_indices[0])
                if len(non_pad_and_non_prefix_indices) > 0
                else 0
            )
            prev_remaining_j = start - 1
            end = (
                int(non_pad_and_non_prefix_indices[-1]) + 1
                if len(non_pad_and_non_prefix_indices) > 0
                else 0
            )
            # Note: We could have used a loop over j going from 0 to max_seq_len,
            # but we don't need to, because we know for sure that there is
            # nothing to count before start and after end. So we save some time.
            # Note: This code should also work if the prefix is non-contiguous.
            for j in range(start, end):
                if _drop_mask[j]:
                    batch_indices.append(e)
                    seq_indices.append(prev_remaining_j)
                    vocab_indices.append(int(example["input_ids"][j].item()))
                    values.append(1)
                else:
                    prev_remaining_j = j

        target_ids = torch.sparse_coo_tensor(
            indices=[batch_indices, seq_indices, vocab_indices],  # type: ignore
            values=values,
            size=(batch_size, max_seq_len, self.vocab_size),
            check_invariants=False,
            is_coalesced=False,
        )
        return {
            "input_ids": collate_tensor_fn(input_ids),
            "attention_mask": collate_tensor_fn(attention_mask),
            "token_type_ids": collate_tensor_fn(token_type_ids),
            "drop": torch.stack(drop_masks, dim=0),
            "target_ids": (
                target_ids.to_dense()
                if self.return_dense_target
                else target_ids
            ),
            "constraint": None,
        }


class NewDefaultILMCollator(Collator):
    """We remove the dropped tokens."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        loss_on_padding: bool = True,
        return_dense_target: bool = False,  # setting to two will increase the cpu memory usage
    ):
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self.loss_on_padding = loss_on_padding
        self.return_dense_target = return_dense_target
        if self.loss_on_padding:
            self.attn_extension_id = 1
        else:
            self.attn_extension_id = 0

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return len(self.tokenizer)

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def sample_n_drops(self, seq_len: int) -> int:
        return np.random.randint(seq_len + 1)

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> NewILMBatch:
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        drop_masks: List[Bool[TT, " max_seq_len"]] = []
        # other tensors
        input_ids: List[Integer[TT, " max_seq_len"]] = []
        attention_mask: List[Integer[TT, " max_seq_len"]] = []
        token_type_ids: List[Integer[TT, " max_seq_len"]] = []
        # Elements of the sparse tensor
        batch_indices: List[int] = []
        seq_indices: List[int] = []
        vocab_indices: List[int] = []
        values: List[int] = []
        for e, _example in enumerate(examples):
            if len(_example["input_ids"]) > max_seq_len:
                raise ValueError(
                    f"Input ids length {len(_example['input_ids'])} is greater than max_seq_len {max_seq_len}"
                )
            example = {
                "input_ids": torch.tensor(
                    _example["input_ids"][:max_seq_len]
                    + (max_seq_len - len(_example["input_ids"]))
                    * [self.tokenizer.pad_token_id]
                ),
                "attention_mask": torch.tensor(
                    _example["attention_mask"][:max_seq_len]
                    + (max_seq_len - len(_example["attention_mask"]))
                    * [self.attn_extension_id]
                ).bool(),
                "token_type_ids": torch.tensor(
                    _example["token_type_ids"][:max_seq_len]
                    + (max_seq_len - len(_example["token_type_ids"])) * [2]
                ),
            }
            non_pad = example["attention_mask"]
            input_ids.append(example["input_ids"])
            attention_mask.append(non_pad)
            token_type_ids.append(example["token_type_ids"])
            non_prefix = (
                example["token_type_ids"] > 1
                if "token_type_ids" in example
                else torch.ones_like(non_pad)
            )
            non_pad_and_non_prefix_indices = torch.logical_and(
                non_pad, non_prefix
            ).nonzero(as_tuple=True)[
                0
            ]  # shape: (*n_non_pad_and_non_prefix,)
            _seq_len = int(len(non_pad_and_non_prefix_indices))
            _n_drops = self.sample_n_drops(_seq_len)
            _pre_drop_indices = (torch.randperm(_seq_len))[
                :_n_drops
            ]  # shape: (n_drops,)
            _drop_indices = non_pad_and_non_prefix_indices[
                _pre_drop_indices
            ]  # shape: (n_drops,)
            _drop_mask = torch.zeros((max_seq_len,), dtype=torch.bool)
            _drop_mask[_drop_indices] = True
            drop_masks.append(_drop_mask)
            # we maintain two indices:
            # 1. The index of the last non-dropped token: prev_remaining_j,
            #   initialized to the index of the token right before the first dropped token
            # 2. The current index: j
            # Check for empty tensor to handle the case blank input during prediction
            start = (
                int(non_pad_and_non_prefix_indices[0])
                if len(non_pad_and_non_prefix_indices) > 0
                else 0
            )
            prev_remaining_j = start - 1
            end = (
                int(non_pad_and_non_prefix_indices[-1]) + 1
                if len(non_pad_and_non_prefix_indices) > 0
                else 0
            )
            # Note: We could have used a loop over j going from 0 to max_seq_len,
            # but we don't need to, because we know for sure that there is
            # nothing to count before start and after end. So we save some time.
            # Note: This code should also work if the prefix is non-contiguous.
            for j in range(start, end):
                if _drop_mask[j]:
                    batch_indices.append(e)
                    seq_indices.append(prev_remaining_j)
                    vocab_indices.append(int(example["input_ids"][j].item()))
                    values.append(1)
                else:
                    prev_remaining_j = j

        target_ids = torch.sparse_coo_tensor(
            indices=[batch_indices, seq_indices, vocab_indices],  # type: ignore
            values=values,
            size=(batch_size, max_seq_len, self.vocab_size),
            check_invariants=False,
            is_coalesced=False,
        )
        return {
            "input_ids": collate_tensor_fn(input_ids),
            "attention_mask": collate_tensor_fn(attention_mask),
            "token_type_ids": collate_tensor_fn(token_type_ids),
            "drop": torch.stack(drop_masks, dim=0),
            "target_ids": (
                target_ids.to_dense()
                if self.return_dense_target
                else target_ids
            ),
            "constraint": None,
        }


class DefaultILMWithLengthClassificationCollator(DefaultILMCollator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        loss_on_padding: bool = False,
        return_dense_target: bool = False,  # setting to two will increase the cpu memory usage
    ):
        if loss_on_padding:
            logger.warning(
                "loss_on_padding is true in collator for length classification setting."
                " This is not the typical setting for ILMWithLengthClassification. "
            )
        super().__init__(
            tokenizer,
            block_size,
            noise_schedule,
            loss_on_padding=loss_on_padding,
            return_dense_target=return_dense_target,
        )


class DefaultILMWithLengthClassificationCollatorForWrappedPaddedSequences(
    DefaultILMWithLengthClassificationCollator
):
    """For pre-padded sequences, that are potentially also wrapped/grouped into blocks.

    Note:
       1. The collator prepares `constraint` tensor, where 1 is set for CLS tokens.
    """

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> ILMBatch:
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        drop_masks: List[Bool[TT, " max_seq_len"]] = []
        # other tensors
        input_ids = collections.deque()
        attention_mask = collections.deque()
        token_type_ids = collections.deque()
        constraint = collections.deque()
        # Elements of the sparse tensor
        batch_indices: List[int] = []
        seq_indices: List[int] = []
        vocab_indices: List[int] = []
        values: List[int] = []
        for e, _example in enumerate(examples):
            non_pad = _example["attention_mask"]
            input_ids.append(_example["input_ids"])
            attention_mask.append(_example["attention_mask"])
            token_type_ids.append(_example["token_type_ids"])
            # _constraint = []  # length: (max_seq_len,)
            non_pad_and_non_cls_non_bos_indices = (
                []
            )  # length: (num_non_pad_and_non_cls,)
            for _i, (_not_pad, _type_id) in enumerate(
                zip(non_pad, _example["token_type_ids"])
            ):
                if _type_id > 1:
                    if _not_pad or self.loss_on_padding:
                        non_pad_and_non_cls_non_bos_indices.append(_i)
                    # _constraint.append(False)
                else:  # _type_id == 0:
                    # _constraint.append(
                    #    True
                    # )  # we don't want token prediction loss on CLS
                    pass
            # constraint.append(_constraint)
            _n_drops = self.sample_n_drops(
                len(non_pad_and_non_cls_non_bos_indices)
            )
            _drop_indices = np.random.default_rng().choice(
                non_pad_and_non_cls_non_bos_indices,
                size=_n_drops,
                replace=False,
            )  # length: (n_drops,)
            _drop_mask = torch.zeros((max_seq_len,), dtype=torch.bool)
            _drop_mask[_drop_indices] = True
            drop_masks.append(_drop_mask)
            # we maintain two indices:
            # 1. The index of the last non-dropped token: prev_remaining_j,
            #   initialized to the index of the token right before the first dropped token
            # 2. The current index: j
            # Check for empty tensor to handle the case blank input during prediction
            start = (
                int(non_pad_and_non_cls_non_bos_indices[0])
                if len(non_pad_and_non_cls_non_bos_indices) > 0
                else 0
            )
            prev_remaining_j = max(
                0, start - 1
            )  # -1 because we want to start at bos
            end = (
                int(non_pad_and_non_cls_non_bos_indices[-1]) + 1
                if len(non_pad_and_non_cls_non_bos_indices) > 0
                else 0
            )
            # Note: We could have used a loop over j going from 0 to max_seq_len,
            # but we don't need to, because we know for sure that there is
            # nothing to count before start and after end. So we save some time.
            # Note: This code should also work if the prefix is non-contiguous.

            # We need to reset when type_id=0 is encountered
            for j in range(start, end):
                # skip if type_id=0, i.e. CLS, assuming that it will be followed by BOS
                if _example["token_type_ids"][j] == 0:
                    prev_remaining_j = (
                        j + 1
                    )  # just to be safe set prev_remaining_j to the next token which is expected to be BOS
                    continue
                if _drop_mask[j]:
                    batch_indices.append(e)
                    seq_indices.append(prev_remaining_j)
                    vocab_indices.append(_example["input_ids"][j])
                    values.append(1)
                else:
                    prev_remaining_j = j

        target_ids = torch.sparse_coo_tensor(
            indices=[batch_indices, seq_indices, vocab_indices],  # type: ignore
            values=values,
            size=(batch_size, max_seq_len, self.vocab_size),
            check_invariants=False,
            is_coalesced=False,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "drop": torch.stack(drop_masks, dim=0),
            "target_ids": (
                target_ids.to_dense()
                if self.return_dense_target
                else target_ids
            ),
            # "constraint": torch.tensor(constraint, dtype=torch.bool),
            "constraint": None,
        }


class DefaultILMCollatorForPrediction(DefaultILMCollator):
    """Drop all dropable tokens."""

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        max_in_batch = max(len(e["input_ids"]) for e in batch)
        return min(max_in_batch, self.block_size)

    def sample_n_drops(self, seq_len: int) -> int:
        return seq_len


class DefaultILMWithLengthClassificationCollatorForPrediction(
    DefaultILMWithLengthClassificationCollator
):
    """Drop all dropable tokens"""

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        max_in_batch = max(len(e["input_ids"]) for e in batch)
        return min(max_in_batch, self.block_size)

    def sample_n_drops(self, seq_len: int) -> int:
        return seq_len


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
    print("attention_mask:")
    print(batch["attention_mask"][0])
    print("token_type_ids:")
    print(batch["token_type_ids"][0])
    print("drop:")
    print(batch["drop"][0])
    print("target_ids:")
    print(
        batch["target_ids"][0].to_sparse()
        if batch is not None and batch.get("target_ids", None) is not None
        else None
    )
    print("constraint:")
    print(batch["constraint"][0] if batch["constraint"] is not None else None)


# endregion: Utilities
################################################################################
