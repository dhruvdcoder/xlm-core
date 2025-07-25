"""Data collation logic for Idlm model.

This file implements the data preprocessing and batching logic for IDLM.
Based on ILM implementation but adapted for IDLM diffusion with noise schedules.
"""

from typing import Dict, Any, Literal
from torch.utils.data import IterableDataset
from xlm.datamodule import Tokenizer


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


# Utility function for debugging
def print_batch_idlm(
    batch: Dict[str, Any],
    split: Literal["train", "val", "test", "predict"],
    tokenizer: Tokenizer,
    dataloader_name: str = "",
):
    """Print batch information for debugging Idlm batches.

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
    print("token_type_ids:")
    print(batch["token_type_ids"][0])
    print("t:")
    print(batch["t"][0])
    if batch["noise_rate"] is not None:
        print("noise_rate:")
        print(batch["noise_rate"][0])
    if batch["total_noise"] is not None:
        print("total_noise:")
        print(batch["total_noise"][0])
    if batch.get("n_drops", None) is not None:
        print("n_drops:")
        print(batch["n_drops"][0])
    print("target_ids (sparse):")
    if hasattr(batch["target_ids"], "to_sparse"):
        print(batch["target_ids"][0].to_sparse())
    else:
        print("(dense target_ids - showing first 5 vocab positions)")
        print(batch["target_ids"][0, :, :5])
    print("constraint:")
    print(batch["constraint"][0] if batch["constraint"] is not None else None)
    print("cls_position:")
    print(
        batch["cls_position"][0] if batch["cls_position"] is not None else None
    )
