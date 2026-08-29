"""Loss function implementation for Bd3lm model.

This file implements the training loss computation. Modify the loss_fn method
to implement your specific loss computation logic.
"""

from typing import Optional
import torch
import torch.nn.functional as F
from xlm.harness import LossFunction, Harness
from xlm.datamodule import Tokenizer
from .types_bd3lm import Bd3lmBatch, Bd3lmLossDict, Bd3lmModel


class Bd3lmLoss(LossFunction[Bd3lmBatch, Bd3lmLossDict]):
    """Loss function for Bd3lm model.
    """

    def __init__(
        self,
        model: Optional[Bd3lmModel] = None,
        tokenizer: Optional[Tokenizer] = None,
        loss_on_padding: bool = True,
    ):
        """Initialize the loss function.

        Args:
            model: The model instance
            tokenizer: Tokenizer for processing tokens
            loss_on_padding: Whether PAD positions on the answer side contribute to
                the loss. 
                True (default): PAD positions count also treated as real tokens in masking and included in the loss.

                False: PAD positions are ingnored and dropped from the loss mask.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.loss_on_padding = loss_on_padding
    def _subs_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity

        logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    
       
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _process_sigma(self, sigma):
        if self.parameterization == 'ar':
            return None
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def loss_fn(
        self,
        batch: Bd3lmBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Bd3lmLossDict:
        """Compute the causal language modeling loss.

        Args:
            batch: The input batch.
            batch_idx: The batch index.
            dataloader_idx: The dataloader index.
            dataloader_name: The dataloader name.

        Returns:
            Dictionary containing the loss.
        """
        x0 = batch["x0"]
        xt = batch["xt"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]
        target_ids = batch["target_ids"]
        loss_scale = batch["loss_scale"]
        sigma = batch["sigma"]
        target_len = target_ids.shape[1]
        assert self.model is not None

        # concateantate noisy and clean prompts

        x_input = torch.cat((xt, x0), dim=-1)
        # Get logits from the model
        sigma = self._process_sigma(sigma)
        
        if bool(attention_mask.all()):
            logits = self.model(x_input, sigma=sigma, attention_mask=None, positions=None)
        else:
            # Create position IDs considering padding on both ends
            # attention_mask has 1 for real tokens, 0 for padding
            positions = attention_mask.cumsum(dim=1) - 1
            positions *= attention_mask  # Zero out positions for padding tokens on the right
            logits = self.model(x_input, sigma=sigma, attention_mask = attention_mask,positions = positions)
        
        model_output = self._subs_parameterization(logits=logits,
                                      xt=xt)
        model_output = model_output[:, -target_len:, :]  # [1, 12, 27]
        loss_mask = loss_mask[:, -target_len:]    # [1, 12]
        ## when loss_on_padding is False, we want to ignore the PAD tokens in the target_ids
        if not self.loss_on_padding and self.tokenizer.pad_token_id is not None:
            loss_mask = loss_mask * (
                target_ids != self.tokenizer.pad_token_id
            ).to(loss_mask.dtype)
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=target_ids[:, :, None]).squeeze(-1)
        
        loss = loss_scale * log_p_theta
        nlls = (loss * loss_mask)
        token_nll = nlls.sum() / loss_mask.sum()
        return {
            "loss": token_nll,
        }
    def __call__(
          self,
          batch: Bd3lmBatch,
          batch_idx: Optional[int] = None,
          dataloader_idx: Optional[int] = None,
          dataloader_name: Optional[str] = None,
      ) -> Bd3lmLossDict:
          return self.loss_fn(batch, batch_idx, dataloader_idx, dataloader_name)

    def configure(self, pl_module: Harness) -> None:
        """Configure the loss function with the lightning module.
        
        This method is called during setup. Use it for any initialization
        that requires the full lightning module.

        Args:
            pl_module: The lightning module instance
        """
        
        self.model = pl_module.model
        self.tokenizer = pl_module.tokenizer
        self.parameterization = self.model.config.algo.parameterization
        self.time_conditioning = self.model.config.algo.time_conditioning
        self.mask_index = self.tokenizer.mask_token_id
        self.neg_infinity = -1000000.0
        
