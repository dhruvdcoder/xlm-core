from typing import (
    Dict,
    List,
    Literal,
    Any,
    Optional,
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
from .types import ZLMBatch, ZLMSeq2SeqBatch

logger = RankedLogger(__name__, rank_zero_only=True)


class ZLMEmptyDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        num_examples: int,
    ):
        """Initialize the ZLM empty dataset.

        Args:
            tokenizer: The tokenizer to use.
            num_examples: Number of empty examples to generate.
        """
        self.tokenizer = tokenizer
        self.num_examples = num_examples

    def __iter__(self):
        """Generate empty examples for ZLM training."""
        for _ in range(self.num_examples):
            ex = self.tokenizer(
                "",
                add_special_tokens=False,
            )
            yield ex


################################################################################
# region: Helper Functions


def prepare_prefix_ids_zlm(
    prefix_ids: List[List[int]],
    pad_token_id: int,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    truncate: Literal["max", "block", None] = "block",
    add_bos: Optional[str] = None,
    add_eos: bool = False,
) -> Dict[str, List[List[int]]]:
    """
    Prepare prefix ids for ZLM seq2seq tasks.

    Args:
        prefix_ids: List of prefix token sequences.
        pad_token_id: Padding token ID.
        bos_token_id: BOS token ID.
        eos_token_id: EOS token ID.
        max_seq_len: Maximum sequence length.
        truncate: Truncation strategy.
        add_bos: Where to add BOS token ("input" for prefix, "output" for after prefix, None for no BOS).
        add_eos: Whether to add EOS token at the end of the prefix.

    Returns:
        Dictionary with input_ids and attention_mask as lists.
    """
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []

    # Determine max length
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
        # Add BOS to prefix if requested
        if add_bos == "input" and bos_token_id is not None:
            temp = [bos_token_id] + _prefix_ids
        elif add_bos == "output" and bos_token_id is not None:
            temp = _prefix_ids + [bos_token_id]  # Add BOS to the right
        else:
            temp = _prefix_ids

        # Add EOS token at the end if requested
        if add_eos and eos_token_id is not None:
            temp = temp + [eos_token_id]

        # Pad/truncate
        padded_seq = pad_truncate_list(
            temp, max_len, pad_token_id, pad_left=True
        )
        input_ids.append(padded_seq)

        # Create attention mask (1 for real tokens, 0 for padding on the left)
        mask = [0] * (max_len - len(temp)) + [1] * len(temp)
        attention_mask.append(mask)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def prepare_suffix_ids_zlm(
    suffix_ids: List[List[int]],
    pad_token_id: int,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    truncate: Literal["max", "block", None] = "block",
    add_bos: Optional[str] = None,
    add_eos: bool = False,
) -> Dict[str, List[List[int]]]:
    """
    Prepare suffix ids for ZLM seq2seq tasks.

    Args:
        suffix_ids: List of suffix token sequences.
        pad_token_id: Padding token ID.
        bos_token_id: BOS token ID.
        eos_token_id: EOS token ID.
        max_seq_len: Maximum sequence length.
        truncate: Truncation strategy.
        add_bos: Where to add BOS token ("input" for prefix, "output" for after prefix, None for no BOS).
        add_eos: Whether to add EOS token at the end of the suffix.

    Returns:
        Dictionary with input_ids, attention_mask, and target_ids as lists.
    """
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    target_ids: List[List[int]] = []

    # Determine max length
    if truncate in ["max", None]:
        max_len = max(len(_suffix_ids) for _suffix_ids in suffix_ids)
        if truncate == "max" and max_seq_len is not None:
            max_len = max(max_len, max_seq_len)
    elif truncate == "block" and max_seq_len is not None:
        max_len = max_seq_len
    else:
        raise ValueError(f"Invalid truncate, max_seq_len: {max_seq_len}")

    assert max_len is not None

    for _suffix_ids in suffix_ids:
        # Add BOS before suffix if requested
        if add_bos == "output" and bos_token_id is not None:
            temp = [bos_token_id] + _suffix_ids
        else:
            temp = _suffix_ids

        # Add EOS token at the end if requested
        if add_eos and eos_token_id is not None:
            temp = temp + [eos_token_id]

        # Pad/truncate
        padded_seq = pad_truncate_list(
            temp, max_len, pad_token_id, pad_left=False
        )
        input_ids.append(padded_seq)

        # Create attention mask
        mask = [1] * len(temp) + [0] * (max_len - len(temp))
        attention_mask.append(mask)

        # Create target_ids (unshifted - will be shifted in collator if needed)
        target_ids.append(padded_seq)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
    }


################################################################################
# region: Collators


class DefaultZLMCollator(Collator):
    """Used for pre-training."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        truncate: Literal["max", "block", None] = "block",
        add_eos: bool = False,
    ):
        """Initialize the ZLM collator.

        Args:
            tokenizer: The tokenizer to use.
            block_size: Maximum sequence length.
            noise_schedule: Noise schedule (not used in ZLM but kept for interface consistency).
            truncate: Truncation strategy.
            add_eos: Whether to add EOS token at the end of the sequence.
        """
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self._vocab_size = len(self.tokenizer)
        self.truncate = truncate
        self.add_eos = add_eos

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
    ) -> ZLMBatch:
        """Collate examples into a batch for ZLM training.

        Args:
            examples: List of examples with input_ids.

        Returns:
            ZLMBatch with input_ids, attention_mask, and target_ids.
        """
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        target_ids: List[List[int]] = []

        # Extract input_ids from examples
        seq_lens = [len(e["input_ids"]) for e in examples]

        # Determine max length based on truncation strategy
        # Account for BOS and EOS tokens that will be added
        tokens_to_add = 1  # BOS token
        if self.add_eos:
            tokens_to_add += 1  # EOS token

        if self.truncate == "max":
            max_len = min(max(seq_lens) + tokens_to_add, self.block_size)
        elif self.truncate == "block":
            max_len = self.block_size
        elif self.truncate is None:
            max_len = max(seq_lens) + tokens_to_add
        else:
            raise ValueError(f"Invalid truncate value: {self.truncate}")

        for example in examples:
            # Get the input sequence
            seq = example["input_ids"]

            # Truncate if necessary (account for BOS and EOS tokens)
            if len(seq) > max_len - tokens_to_add:
                seq = seq[: max_len - tokens_to_add]

            # Add BOS token at the beginning
            seq_with_bos = [self.tokenizer.bos_token_id] + seq

            # Add EOS token at the end if requested
            if self.add_eos:
                seq_with_bos = seq_with_bos + [self.tokenizer.eos_token_id]

            # Pad to max_len
            padded_seq = pad_truncate_list(
                seq_with_bos,
                max_len,
                self.tokenizer.pad_token_id,
                pad_left=False,
            )
            input_ids.append(padded_seq)

            # Create attention mask (1 for real tokens including BOS/EOS, 0 for padding)
            mask = [1] * len(seq_with_bos) + [0] * (
                max_len - len(seq_with_bos)
            )
            attention_mask.append(mask)

            # Create target_ids (shifted by 1 for next token prediction)
            # For ZLM, target_ids are the same as input_ids but shifted left by 1
            # Use -100 for padding positions to ignore them during loss computation
            target_seq = seq_with_bos[1:] + [-100]  # Shift left by 1
            # Set padding positions to -100
            for j in range(len(target_seq)):
                if (
                    j < len(mask) - 1 and mask[j + 1] == 0
                ):  # Check if next position is padding
                    target_seq[j] = -100

            target_ids.append(target_seq)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }


class ZLMSeq2SeqCollator:

    def __init__(
        self,
        tokenizer: Tokenizer,
        noise_schedule: NoiseSchedule,
        block_size: Optional[int] = None,
        input_block_size: Optional[int] = None,
        add_bos: Optional[str] = None,
        add_eos: bool = False,
        truncate: Literal["max", "block", None] = "block",
    ):
        """Initialize the ZLM sequence-to-sequence collator.

        Args:
            tokenizer: The tokenizer to use.
            noise_schedule: Noise schedule (not used in ZLM but kept for interface consistency).
            block_size: Maximum sequence length for the target.
            input_block_size: Maximum sequence length for the input.
            add_bos: Where to add BOS token ("input" for prefix, "output" for after prefix, None for no BOS).
            add_eos: Whether to add EOS token at the end of the suffix.
            truncate: Truncation strategy.
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.input_block_size = input_block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.truncate = truncate
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
    ) -> ZLMSeq2SeqBatch:
        """Collate examples into a batch for ZLM sequence-to-sequence training.

        Args:
            examples: List of examples with prompt_ids and input_ids.

        Returns:
            ZLMSeq2SeqBatch with input_ids, attention_mask, target_ids.
        """
        # Prepare prefix (prompt)
        prefix = prepare_prefix_ids_zlm(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=self.input_block_size,
            truncate=self.truncate,
            add_bos=self.add_bos,
            add_eos=False,  # No EOS in prefix for seq2seq
        )

        # Prepare suffix (target)
        suffix = prepare_suffix_ids_zlm(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=self.block_size,
            truncate=self.truncate,
            add_bos=None,  # BOS through prefix
            add_eos=self.add_eos,
        )

        # Concatenate prefix and suffix as lists
        input_ids = [
            p + s for p, s in zip(prefix["input_ids"], suffix["input_ids"])
        ]
        attention_mask = [
            p + s
            for p, s in zip(prefix["attention_mask"], suffix["attention_mask"])
        ]

        # Create target_ids (shifted by 1 for next token prediction)
        # For ZLM seq2seq, target_ids are shifted left by 1
        target_ids = []
        for i, (input_seq, mask) in enumerate(zip(input_ids, attention_mask)):
            target_seq = input_seq[1:] + [-100]  # Shift left by 1
            # Set padding positions to -100
            for j in range(len(target_seq)):
                if (
                    j < len(mask) - 1 and mask[j + 1] == 0
                ):  # Check if next position is padding
                    target_seq[j] = -100
            target_ids.append(target_seq)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "token_type_ids": torch.zeros(
                len(input_ids),
                max(len(seq) for seq in input_ids),
                dtype=torch.long,
            ),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }


class ZLMSeq2SeqPredCollator(ZLMSeq2SeqCollator):
    """Drops all the suffix/target tokens and sends them in the target_ids of shape (batch_size, target_seq_len)"""

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> ZLMSeq2SeqBatch:
        """Collate examples into a batch for ZLM sequence-to-sequence prediction.

        Args:
            examples: List of examples with prompt_ids and input_ids.

        Returns:
            ZLMSeq2SeqBatch with input_ids, attention_mask, target_ids.
        """
        # For prediction, we only need the prefix (prompt) and the target_ids
        # Prepare prefix (prompt)
        prefix = prepare_prefix_ids_zlm(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=self.input_block_size,
            truncate=self.truncate,
            add_bos=self.add_bos,
            add_eos=False,  # No EOS in prefix for seq2seq
        )

        # Prepare target_ids (the full suffix sequence)
        target_ids = prepare_suffix_ids_zlm(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=self.block_size,
            truncate=self.truncate,
            add_bos=None,
            add_eos=self.add_eos,
        )

        # For prediction, input_ids is just the prefix
        input_ids = prefix["input_ids"]
        attention_mask = prefix["attention_mask"]

        # target_ids is the full suffix sequence (not shifted)
        target_ids = target_ids[
            "target_ids"
        ]  # Use unshifted target_ids for prediction

        # Convert to tensors
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "token_type_ids": torch.zeros(
                len(input_ids),
                max(len(seq) for seq in input_ids),
                dtype=torch.long,
            ),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }


# endregion: Collators
################################################################################


################################################################################
# region: Utilities


def _replace_100_with_pad(ids: torch.Tensor, tokenizer: Tokenizer):
    _ids = ids.clone()
    _ids[_ids == -100] = tokenizer.pad_token_id
    return _ids


def print_batch_zlm(
    batch: Dict[str, Any],
    split: Literal["train", "val", "test", "predict"],
    tokenizer: Tokenizer,
    dataloader_name: str = "",
):
    """Print batch information for debugging ZLM batches.

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


# endregion: Utilities
################################################################################
