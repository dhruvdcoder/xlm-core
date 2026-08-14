""" Data Collators for Bd3lm.

This file implements the batching and noise scheduling logic.
"""

from typing import List, Dict, Any, Optional, Literal
import torch
from torch.utils.data import IterableDataset
from xlm.datamodule import Collator, Tokenizer, Seq2SeqCollatorInput, BaseCollatorInput
from xlm.noise import NoiseSchedule
from xlm.utils.nn import pad_truncate_list
from .types_bd3lm import Bd3lmBatch, Bd3lmSeq2SeqBatch

class Bd3lmEmptyDataset(IterableDataset):
    """ This will construct empty rows to drive the prediction loop when generating unconditionally.
      as unconditinal generation has no prompt, we need to create a dummy dataset to drive the prediction loop."""

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
            yield {"input_ids": []}


class Bd3lmUnconditionalPredCollator(Collator):
    """Turns the empty rows into an all-[MASK] sequence for the sampler to fill in.
        ALl the sequences are of the same length with Masks, so the sampler can just fill in the whole sequence with noise and then denoise it."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        max_length: int,
        add_bos: bool = True,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_length = max_length
        self.add_bos = add_bos

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.max_length

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> Dict[str, Any]:
        batch_size = len(examples)
        input_ids = torch.full(
            (batch_size, self.max_length),
            int(self.tokenizer.mask_token_id),
            dtype=torch.long,
        )
        if self.add_bos:
            input_ids[:, 0] = int(self.tokenizer.bos_token_id)
        return {
            "input_ids": input_ids,
            # the canvas is full width, so nothing is padded
            "attention_mask": torch.ones(
                (batch_size, self.max_length), dtype=torch.bool),
        }


class DefaultBd3lmCollator(Collator):
    """Pre-training collator: noises the whole sequence.

    Bd3lmSeq2SeqCollator is the counterpart that keeps the prompt and only noises the answer.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        max_length: Optional[int] = None,
        truncate: Literal["max", "block", None] = "block",
        add_eos: bool = False,
        loss_on_padding: bool = True,
        ignore_bos: bool = True,
    ):
        
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self._vocab_size = len(self.tokenizer)
        self.truncate = truncate
        self.add_eos = add_eos
        self.max_length = max_length if max_length is not None else block_size
        self.loss_on_padding = loss_on_padding
        self.ignore_bos = ignore_bos
        if self.max_length % self.block_size != 0:
            raise ValueError(
                f"max_length ({self.max_length}) must be a multiple of block_size "
                f"({self.block_size}): the noise schedule builds t as "
                f"(max_length // block_size) blocks repeated block_size times, so a "
                f"non-multiple gives a t narrower than x0 and q_xt cannot broadcast.")

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not set")
            self._vocab_size = len(self.tokenizer)
        return self._vocab_size

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.max_length

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> Bd3lmBatch:
        """Build x0, noise it into xt, and mark which positions the loss should score."""
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        target_ids: List[List[int]] = []

        seq_lens = [len(e["input_ids"]) for e in examples]

        # leave room for the special tokens we are about to add
        tokens_to_add = 1  # BOS
        if self.add_eos:
            tokens_to_add += 1  # EOS

        if self.truncate == "max":
            max_len = min(max(seq_lens) + tokens_to_add, self.max_length)
        elif self.truncate == "block":
            max_len = self.max_length
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

            
            if len(seq_with_bos) < max_len and self.tokenizer.pad_token_id is None:
                raise ValueError(
                    f"a sequence is {len(seq_with_bos)} tokens and needs padding to "
                    f"{max_len}, but the tokenizer has no pad_token. Add one via the "
                    f"tokenizer's special_tokens, e.g. pad_token: \"<|endoftext|>\"."
                )
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

            # target_ids are the clean sequence.
            target_ids.append(padded_seq)

        x0 = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.bool)

        t = self.noise_schedule._sample_t(x0.shape, x0.device)
        loss_scale, p = self.noise_schedule(t)
        sigma = self.noise_schedule._sigma_from_p(p[:, 0].unsqueeze(-1))
        xt = self.noise_schedule.q_xt(
            x0,
            p,
            mask_token_id=self.tokenizer.mask_token_id,
            # see loss_on_padding in __init__
            pad_token_id=None if self.loss_on_padding else self.tokenizer.pad_token_id,
        )

       
        if self.ignore_bos:
            xt[:, 0] = x0[:, 0]

        
        loss_mask = (xt == self.tokenizer.mask_token_id).long()
        if self.ignore_bos:
            loss_mask[:, 0] = 0
        if not self.loss_on_padding:
            loss_mask = loss_mask * attention_mask_t.long()

        return {
            "xt": xt,
            "x0": x0,
            "attention_mask": attention_mask_t,
            "loss_mask": loss_mask,
            "token_type_ids": torch.zeros_like(x0),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "loss_scale": loss_scale,
            "sigma": sigma,
            "input_ids": torch.cat((xt, x0), dim=-1),
        }


################################################################################
# region: Helper Functions


def prepare_prefix_ids_bd3lm(
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
    Prepare prefix ids for Bd3lm seq2seq tasks.

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
        # Add BOS to prefix 
        if add_bos == "input" and bos_token_id is not None:
            temp = [bos_token_id] + _prefix_ids
        elif add_bos == "output" and bos_token_id is not None:
            temp = _prefix_ids + [bos_token_id]  # Add BOS to the right
        else:
            temp = _prefix_ids
        # Add EOS token at the end 
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


def prepare_suffix_ids_bd3lm(
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
    Prepare suffix ids for Bd3lm seq2seq tasks.

    Args:
        suffix_ids: List of suffix token sequences.
        pad_token_id: Padding token ID.
        bos_token_id: BOS token ID.
        eos_token_id: EOS token ID.
        max_seq_len: Maximum sequence length.
        truncate: Truncation strategy.
        add_bos: Where to add BOS token 
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
        # Add BOS before suffix
        if add_bos == "output" and bos_token_id is not None:
            temp = [bos_token_id] + _suffix_ids
        else:
            temp = _suffix_ids

        # Add EOS token at the end 
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
        
        target_ids.append(padded_seq)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
    }


################################################################################
# region: Collators


class Bd3lmSeq2SeqCollator:
    """Seq2seq collator for Bd3lm model.
    
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        noise_schedule: NoiseSchedule,
        block_size: Optional[int] = None,
        input_block_size: Optional[int] = None,
        add_bos: Optional[str] = None,
        add_eos: bool = False,
        truncate: Literal["max", "block", None] = "block",
        prompt_size: Optional[int] = None,
        target_size: Optional[int] = None,
        loss_on_padding: bool = True,

    ):
        # input_block_size is the prompt width used by Bd3lmSeq2SeqPredCollator, which
        # inherits this __init__ - training uses prompt_size instead, so it is None here.
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.input_block_size = input_block_size
        self.noise_schedule = noise_schedule
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.truncate = truncate
        self._vocab_size = (
            len(self.tokenizer) if self.tokenizer is not None else None
        )
        self.prompt_size = prompt_size
        self.target_size = target_size
        self.loss_on_padding = loss_on_padding

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
    ) -> Bd3lmSeq2SeqBatch:
        """Collate examples into a batch for Bd3lm sequence-to-sequence training.

        Args:
            examples: List of examples with prompt_ids and input_ids.

        Returns:
            Bd3lmSeq2SeqBatch with input_ids, attention_mask, target_ids.
        """
        # Prepare prefix (prompt)
        prefix = prepare_prefix_ids_bd3lm(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=self.prompt_size,
            truncate=self.truncate,
            add_bos=self.add_bos,
            add_eos=False,# No EOS in prefix for seq2seq
        )

        # Prepare suffix (target) 
        suffix = prepare_suffix_ids_bd3lm(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=self.target_size,
            truncate=self.truncate,
            add_bos=None,  # BOS through prefix
            add_eos=self.add_eos,
        )
        ## noisy masking for the padded clean suffix....

        clean_suffix = suffix["input_ids"]
        suffix_attention_mask = suffix["attention_mask"]
        clean_suffix = torch.tensor(clean_suffix)
        ## Masking the clean suffix...
        t = self.noise_schedule._sample_t(
            clean_suffix.shape,
            clean_suffix.device,
        )
        loss_scale, p = self.noise_schedule(t)
        sigma = self.noise_schedule._sigma_from_p(p[:,0].unsqueeze(-1))
        # loss_on_padding=True keeps pad_token_id=None, which lets q_xt noise PAD
        # into MASK like any other token. False passes the id through, and q_xt
        # then excludes those positions.
        noisy_suffix = self.noise_schedule.q_xt(
            clean_suffix,
            p,
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=None if self.loss_on_padding else self.tokenizer.pad_token_id,
        )
        if self.noise_schedule.sampling_eps_min is not None and self.noise_schedule.sampling_eps_min > 0.5:
            loss_scale = -torch.ones_like(loss_scale)
        
        ## concatenate prefix and noisy suffix 
        prefix_input_ids = torch.tensor(prefix["input_ids"], dtype=torch.long)
        
        xt = torch.cat([prefix_input_ids, noisy_suffix], dim=-1)

        # Concatenate prefix and clean suffix as lists
        x0 = [
            p + s for p, s in zip(prefix["input_ids"], suffix["input_ids"])
        ]
        suffix_full_attention = [[1] * len(s) for s in suffix["attention_mask"]]
        attention_mask = [ p + s for p, s in zip(prefix["attention_mask"], suffix_full_attention) ] 

        target_ids = clean_suffix
        ### loss mask...
        prefix_attention_mask = torch.tensor(prefix["attention_mask"], dtype=torch.long)
        prefix_loss_mask = torch.zeros_like(prefix_attention_mask)
        suffix_loss_mask = (noisy_suffix == self.tokenizer.mask_token_id).long()
        if not self.loss_on_padding:
           
            suffix_loss_mask = suffix_loss_mask * torch.tensor(
                suffix_attention_mask, dtype=suffix_loss_mask.dtype
            )

        loss_mask = torch.cat([prefix_loss_mask, suffix_loss_mask], dim=-1)
        
        x0 = torch.tensor(x0, dtype=torch.long)
        return {
            "xt": xt,
            "x0": x0,
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "loss_mask": loss_mask,
            "token_type_ids": torch.zeros(
                len(x0),
                max(len(seq) for seq in x0),
                dtype=torch.long,
            ),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "loss_scale": loss_scale,
            "sigma": sigma,
            "input_ids": torch.cat((xt, x0), dim=-1),
        }
class Bd3lmSeq2SeqPredCollator(Bd3lmSeq2SeqCollator):
    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> Bd3lmSeq2SeqBatch:
        """Collate examples into a batch for Bd3lm sequence-to-sequence prediction.

        Args:
            examples: List of examples with prompt_ids and input_ids.

        Returns:
            Bd3lmSeq2SeqBatch with input_ids, attention_mask, target_ids.
        """
        
        # For prediction, we only need the prefix (prompt) and target_ids
        # Prepare prefix (prompt)
        prefix = prepare_prefix_ids_bd3lm(
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
        current_length = 0
        for e in examples:
            current_length = max(current_length,len(e["input_ids"]))
        
        target_ids = prepare_suffix_ids_bd3lm(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=current_length+1,
            truncate=self.truncate,
            add_bos=None,
            add_eos=True,
        )

        # For prediction, input_ids is just the prefix
        input_ids = prefix["input_ids"]
        attention_mask = prefix["attention_mask"]

        # target_ids is the ground truth sequence.
        target_ids = target_ids[
            "target_ids"
        ]  
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


def print_batch_bd3lm(
    batch: Dict[str, Any],
    split: Literal["train", "val", "test", "predict"],
    tokenizer: Tokenizer,
    dataloader_name: str = "",
):
    """Print batch information for debugging Bd3lm batches.

    Args:
        batch: The batch to print.
        split: The split name.
        tokenizer: The tokenizer to decode tokens.
        dataloader_name: Name of the dataloader.
    """
    print(
        f"Printing first entries of the tensors in batch for {split}/{dataloader_name}..."
    )
    print("input tokens:")
    print(tokenizer.decode(batch["input_ids"][0]))
    print("input_ids:")
    print(batch["input_ids"][0])
    print("attention_mask (int):")
    print(batch["attention_mask"][0].int())
    
    if "target_ids" in batch:
        print("target_ids:")
        print(batch["target_ids"][0])
        print("target tokens:")
        print(tokenizer.decode(batch["target_ids"][0]))
    else:
        print("target_ids: (none - unconditional batch)")


# endregion: Utilities
################################################################################
