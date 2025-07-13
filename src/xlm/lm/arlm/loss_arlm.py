from typing import (
    Optional,
)

import torch

from xlm.harness import (
    Harness,
    LossFunction,
)
from xlm.utils.rank_zero import RankedLogger
from .types_arlm import ARLMBatch, ARLMLossDict, ARLMModel
from xlm.datamodule import Tokenizer

logger = RankedLogger(__name__, rank_zero_only=True)


###############################################################
# region: Loss functions


class ARLMLoss(LossFunction[ARLMBatch, ARLMLossDict]):
    """Loss function for Auto-Regressive Language Modeling (ARLM).

    This loss function implements causal language modeling where the model predicts
    the next token given the previous tokens. The loss is computed using cross-entropy
    on the target sequence (which is already shifted in the batch).

    For seq2seq tasks, loss is only computed on suffix tokens (non-prompt tokens).
    """

    def __init__(
        self,
        model: Optional[ARLMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """Initialize the ARLM loss function.

        Args:
            model: The ARLM model to use for predictions.
            tokenizer: The tokenizer for processing tokens.
        """
        self.model = model
        self.tokenizer = tokenizer

    def configure(self, pl_module: Harness):
        """Configure the loss function with the lightning module.

        Args:
            pl_module: The lightning module instance.
        """
        pass  # nothing to configure for ARLM

    def __call__(
        self,
        batch: ARLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ARLMLossDict:
        """Compute the loss for the given batch.

        Args:
            batch: The input batch containing input_ids, attention_mask, and target_ids.
            batch_idx: The batch index.
            dataloader_idx: The dataloader index.
            dataloader_name: The dataloader name.

        Returns:
            Dictionary containing the loss, batch_loss, and nlls.
        """
        return self.loss_fn(batch, batch_idx, dataloader_idx, dataloader_name)

    def loss_fn(
        self,
        batch: ARLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ARLMLossDict:
        """Compute the causal language modeling loss.

        Args:
            batch: The input batch.
            batch_idx: The batch index.
            dataloader_idx: The dataloader index.
            dataloader_name: The dataloader name.

        Returns:
            Dictionary containing the loss, batch_loss, and nlls.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_ids = batch["target_ids"]

        assert self.model is not None

        # Create position IDs considering padding on both ends
        # attention_mask has 1 for real tokens, 0 for padding
        positions = attention_mask.cumsum(dim=1) - 1
        positions *= attention_mask  # Zero out positions for padding tokens on the right. Not strictly necessary because attention_mask is already 0 for padding tokens.

        # Create causal attention mask with padding consideration
        _, seq_len = attention_mask.shape
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=attention_mask.device,
            )
        )

        # Expand attention_mask to (batch, seq_len, seq_len) and combine with causal mask
        expanded_attention_mask = attention_mask.unsqueeze(
            1
        )  # (batch, 1, seq_len)
        causal_attention_mask = (
            expanded_attention_mask & causal_mask
        )  # (batch, seq_len, seq_len)

        # Get logits from the model
        logits = self.model(input_ids, causal_attention_mask, positions)

        # For causal LM, we predict the next token
        # Since target_ids are already shifted we don't need to shift again
        # We don't even need to remove the last token from logits

        # Transpose logits for cross_entropy (expects [N, C, ...] format)
        logits_T = logits.transpose(1, 2)

        # Compute cross-entropy loss ( ignores -100 positions)
        ce_loss = torch.nn.functional.cross_entropy(
            logits_T, target_ids, reduction="mean", ignore_index=-100
        )  # (batch, seq_len)

        return {
            "loss": ce_loss,
        }
