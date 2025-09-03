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
from xlm.utils.nn import pad_truncate_list
from .types_indigo import IndigoBatch

logger = RankedLogger(__name__, rank_zero_only=True)


class IndigoEmptyDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        num_examples: int,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples

    def __iter__(self):
        # TODO (URV)
        pass


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
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self._vocab_size = len(self.tokenizer)
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
    ) -> Dict[str, Any]:
        # TODO (URV): Implement the collator.
        pass


class IndigoSeq2SeqCollator:
    """Drops tokens from the suffix only."""

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
    ) -> Dict[str, Any]:
        # TODO (URV): Implement the collator.
        pass


class IndigoSeq2SeqPredCollator(IndigoSeq2SeqCollator):
    """Drops all the suffix/target tokens and sends them in the target_ids of shape (batch_size, target_seq_len)"""

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> Dict[str, Any]:
        # TODO (URV): Implement the collator.
        pass


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
    # TODO (URV)
    pass


# endregion: Utilities
################################################################################
