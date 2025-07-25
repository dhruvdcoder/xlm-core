"""Data collation logic for Idlm model.

This file implements the data preprocessing and batching logic for IDLM.
Based on ILM implementation but adapted for IDLM diffusion with noise schedules.
"""

from typing import List, Dict, Any, Optional, Literal, Callable
import numpy as np
import torch
from torch import Tensor as TT
from torch.utils.data import IterableDataset
from xlm.datamodule import Collator, Tokenizer, Seq2SeqCollatorInput
from xlm.noise import NoiseSchedule
from xlm.utils.nn import pad_truncate_list
from .types import IdlmBatch


class IdlmEmptyDataset(IterableDataset):
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


def _drop_uniformly(seq_len: int, n_drops: int) -> List[int]:
    """Sample n_drops indices uniformly from [0, seq_len)."""
    return np.random.permutation(seq_len)[:n_drops].tolist()


def _sample_n_drops_from_noise_schedule(
    noise_schedule: NoiseSchedule, t: float, seq_len: int
) -> int:
    """Sample number of drops from noise schedule given time t."""
    t_tensor = torch.tensor([t])
    noise_rate, total_noise = noise_schedule(t_tensor)
    # Sample from Poisson distribution with rate total_noise
    n_drops = int(torch.poisson(total_noise).item())
    # Clamp to valid range
    return min(max(0, n_drops), seq_len)


def idlm_drop_fn(
    segment_input_ids: List[int],
    bos_token_id: int,
    cls_token_id: Optional[int],
    noise_schedule: NoiseSchedule,
    t: float,
    global_offset: int = 0,
    drop_indices_fn: Callable[[int, int], List[int]] = _drop_uniformly,
) -> Dict[str, Any]:
    """Drops tokens from a single segment using IDLM noise schedule. Adds bos. Adds cls as requested."""
    _input_ids = segment_input_ids
    offset = 2 if cls_token_id is not None else 1
    target_seq_indices: List[int] = []
    target_vocab_indices: List[int] = []
    target_values: List[int] = []
    _orig_seq_len = len(segment_input_ids)

    # Use noise schedule to determine number of drops
    _n_drops = _sample_n_drops_from_noise_schedule(
        noise_schedule, t, _orig_seq_len
    )
    _drop_indices = drop_indices_fn(_orig_seq_len, _n_drops)
    _drop_indices_set = set(_drop_indices)

    _input_ids_with_drops: List[int] = (
        [cls_token_id, bos_token_id] + [None] * (_orig_seq_len - _n_drops)  # type: ignore
        if cls_token_id is not None
        else [bos_token_id] + [None] * (_orig_seq_len - _n_drops)  # type: ignore
    )

    i = offset - 1  # index in the post-drop sequence

    # Process each token in original sequence
    for j in range(_orig_seq_len):
        if j in _drop_indices_set:
            # This token is dropped - add to targets
            target_seq_indices.append(global_offset + i)
            target_vocab_indices.append(int(_input_ids[j]))
            target_values.append(1)
        else:
            # This token is kept - add to sequence
            _input_ids_with_drops[i + 1] = _input_ids[j]
            i += 1

    # Validate that all positions are filled
    assert not any(_idx is None for _idx in _input_ids_with_drops)

    return {
        "segment_input_ids_with_drops": _input_ids_with_drops,
        "segment_target_seq_indices": target_seq_indices,
        "segment_target_vocab_indices": target_vocab_indices,
        "segment_target_values": target_values,
        "segment_n_drops": _n_drops,
    }


def prepare_prefix_ids_idlm(
    prefix_ids: List[List[int]],
    pad_token_id: int,
    max_seq_len: Optional[int] = None,
    cls_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    bos_side: Literal["left", "right"] = "right",
) -> Dict[str, TT]:
    """Prepare prefix IDs for IDLM seq2seq tasks."""
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

        padded, num_padded = pad_truncate_list(
            temp,
            max_len,
            pad_token_id,
            pad_left=True,
            return_num_padded=True,
        )
        cls_positions.append(num_padded)
        input_ids.append(padded)

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


def idlm_single_segment_collate_target_fn(
    examples: List[List[int]],
    pad_token_id: int,
    bos_token_id: int,
    vocab_size: int,
    cls_token_id: Optional[int],
    noise_schedule: NoiseSchedule,
    type_extension_id: int = 2,
    pad_left: bool = False,
    max_seq_len: Optional[int] = None,
    truncate: Literal["max", "block", None] = "block",
    global_offset: int = 0,
    return_dense_target: bool = False,
    return_dense_n_drops: bool = True,
    drop_indices_fn: Callable[[int, int], List[int]] = _drop_uniformly,
) -> IdlmBatch:
    """Collate single segment examples for IDLM with diffusion noise schedule."""
    # Sample time steps for this batch
    batch_size = len(examples)
    t_samples = noise_schedule.sample_t(batch_size).tolist()
    noise_rates = []
    total_noises = []

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

    # First pass: process each example and determine max length
    for e, (_example, t) in enumerate(zip(examples, t_samples)):
        # Get noise parameters for this time step
        t_tensor = torch.tensor([t])
        noise_rate, total_noise = noise_schedule(t_tensor)
        noise_rates.append(noise_rate.item())
        total_noises.append(total_noise.item())

        # Process this example with dropping
        single_seq_drop_result = idlm_drop_fn(
            _example,
            bos_token_id,
            cls_token_id,
            noise_schedule,
            t,
            global_offset,
            drop_indices_fn=drop_indices_fn,
        )

        # Add batch index
        single_seq_drop_result["segment_target_batch_indices"] = [e] * len(
            single_seq_drop_result["segment_target_seq_indices"]
        )

        max_seq_len_in_batch = max(
            max_seq_len_in_batch,
            len(single_seq_drop_result["segment_input_ids_with_drops"]),
        )
        results.append(single_seq_drop_result)

    # Determine final max length
    if truncate == "max":
        max_len = max_seq_len_in_batch
    elif truncate == "block":
        if max_seq_len is None:
            raise ValueError(
                "max_seq_len must be provided when truncate='block'"
            )
        max_len = max_seq_len
    elif truncate is None:
        max_len = max_seq_len_in_batch
    else:
        raise ValueError(f"Invalid truncate value: {truncate}")

    # Second pass: pad and create final tensors
    for e, single_seq_drop_result in enumerate(results):
        # Store results for sparse tensors
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

        # Add n_drops information
        n_drops_count = single_seq_drop_result["segment_n_drops"]
        for seq_idx in single_seq_drop_result["segment_target_seq_indices"]:
            n_drops_batch_indices.append(e)
            n_drops_seq_indices.append(seq_idx)
            n_drops_values.append(1)

        # Pad and prepare final batch tensors
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

        # Token type IDs: 0 for CLS, 1 for prefix (fixed), 2 for non-prefix tokens
        token_type_ids.append(
            pad_truncate_list(
                (
                    [0, 1]
                    + [type_extension_id]
                    * (
                        len(
                            single_seq_drop_result[
                                "segment_input_ids_with_drops"
                            ]
                        )
                        - 2
                    )
                    if cls_token_id is not None
                    else [1]  # BOS
                    + [type_extension_id]
                    * (
                        len(
                            single_seq_drop_result[
                                "segment_input_ids_with_drops"
                            ]
                        )
                        - 1
                    )
                ),
                max_len,
                type_extension_id,
                pad_left,
            )
        )

    # Create sparse tensor for targets
    target_ids = torch.sparse_coo_tensor(
        indices=[  # type: ignore
            target_batch_indices,
            target_seq_indices,
            target_vocab_indices,
        ],
        values=target_values,
        size=(batch_size, global_offset + max_len, vocab_size),
        check_invariants=False,
        is_coalesced=False,
    )

    # Create sparse tensor for n_drops (similar to ILM)
    n_drops = torch.sparse_coo_tensor(
        indices=[  # type: ignore
            n_drops_batch_indices,
            n_drops_seq_indices,
        ],
        values=n_drops_values,
        size=(batch_size, global_offset + max_len),
        check_invariants=False,
        is_coalesced=False,
    )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "target_ids": (
            target_ids.to_dense() if return_dense_target else target_ids
        ),
        "n_drops": (n_drops.to_dense() if return_dense_n_drops else n_drops),
        "t": torch.tensor(t_samples, dtype=torch.float),
        "noise_rate": torch.tensor(noise_rates, dtype=torch.float),
        "total_noise": torch.tensor(total_noises, dtype=torch.float),
        "constraint": None,
        "cls_position": torch.zeros(len(input_ids), dtype=torch.long),
    }


class DefaultIdlmCollator(Collator):
    """Default collator for Idlm model.

    Used for pre-training with diffusion noise schedules.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        truncate: Literal["max", "block", None] = "block",
        return_dense_target: bool = False,
    ):
        """Initialize the Idlm collator.

        Args:
            tokenizer: The tokenizer to use.
            block_size: Maximum sequence length.
            noise_schedule: Noise schedule for diffusion.
            truncate: Truncation strategy.
            return_dense_target: Whether to return dense target tensor.
        """
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self._vocab_size = len(self.tokenizer)
        self.truncate = truncate
        self.return_dense_target = return_dense_target

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not set")
            self._vocab_size = len(self.tokenizer)
        return self._vocab_size

    def get_max_len(self, batch: List[Dict[str, Any]]) -> int:
        return self.block_size

    def __call__(
        self,
        examples: List[Dict[str, Any]],
    ) -> IdlmBatch:
        """Collate examples into a batch for Idlm training.

        Args:
            examples: List of examples with input_ids.

        Returns:
            IdlmBatch with diffusion-specific fields.
        """
        batch = idlm_single_segment_collate_target_fn(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.vocab_size,
            self.tokenizer.cls_token_id,
            self.noise_schedule,
            type_extension_id=2,
            pad_left=False,
            max_seq_len=self.block_size,
            truncate=self.truncate,
            global_offset=0,
            return_dense_target=self.return_dense_target,
            drop_indices_fn=_drop_uniformly,
        )
        cls_position = torch.zeros(
            batch["input_ids"].shape[0], dtype=torch.long
        )
        return {
            **batch,
            "cls_position": cls_position,
        }


class IdlmSeq2SeqCollator:
    """Seq2seq collator for Idlm model.

    Drops tokens from the suffix only using diffusion noise schedule.
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
    ):
        """Initialize the Idlm sequence-to-sequence collator.

        Args:
            tokenizer: The tokenizer to use.
            noise_schedule: Noise schedule for diffusion.
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
    ) -> IdlmBatch:
        """Collate examples into a batch for Idlm sequence-to-sequence training.

        Args:
            examples: List of examples with prompt_ids and input_ids.

        Returns:
            IdlmBatch with diffusion-specific fields.
        """
        # Handle the prefix
        cls_token_id = self.tokenizer.cls_token_id
        prefix = prepare_prefix_ids_idlm(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            max_seq_len=self.input_block_size,
            cls_token_id=cls_token_id,
        )
        global_offset = prefix["input_ids"].shape[1]

        # Handle the suffix with IDLM dropping
        suffix = idlm_single_segment_collate_target_fn(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.vocab_size,
            cls_token_id=None,  # Already added to prefix
            noise_schedule=self.noise_schedule,
            type_extension_id=2,
            pad_left=False,
            max_seq_len=self.block_size,
            global_offset=global_offset,
            return_dense_target=False,
            drop_indices_fn=_drop_uniformly,
        )

        # Concatenate prefix and suffix
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
            "t": suffix["t"],
            "noise_rate": suffix["noise_rate"],
            "total_noise": suffix["total_noise"],
            "constraint": None,
            "cls_position": prefix["cls_position"],
        }


class IdlmSeq2SeqPredCollator(IdlmSeq2SeqCollator):
    """Prediction collator for Idlm sequence-to-sequence tasks.

    Uses prefix only, sends targets for evaluation.
    """

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> IdlmBatch:
        """Collate examples into a batch for Idlm sequence-to-sequence prediction.

        Args:
            examples: List of examples with prompt_ids and input_ids.

        Returns:
            IdlmBatch prepared for prediction.
        """
        # Handle the prefix
        cls_token_id = self.tokenizer.cls_token_id
        prefix = prepare_prefix_ids_idlm(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            max_seq_len=self.input_block_size,
            cls_token_id=cls_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        # Prepare target_ids for evaluation (no dropping)
        target_ids = []
        max_target_len = (
            max(len(e["input_ids"]) for e in examples) if examples else 0
        )
        if self.block_size:
            max_target_len = min(max_target_len, self.block_size)

        for e in examples:
            padded_target = pad_truncate_list(
                e["input_ids"],
                max_target_len,
                self.tokenizer.pad_token_id,
                pad_left=False,
            )
            target_ids.append(padded_target)

        batch_size = prefix["input_ids"].shape[0]
        constraint = torch.ones_like(prefix["input_ids"], dtype=torch.bool)
        constraint[:, -1] = 0

        return {
            "input_ids": prefix["input_ids"],
            "attention_mask": prefix["attention_mask"],
            "token_type_ids": prefix["token_type_ids"],
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "n_drops": None,
            "t": torch.ones(batch_size),
            "noise_rate": None,
            "total_noise": None,
            "constraint": constraint,
            "cls_position": prefix["cls_position"],
        }
