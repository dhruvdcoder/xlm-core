from typing import Optional, cast

import torch
from .types_mlm import MLMBatch, MLMLossDict, MLMModel
from xlm.harness import LossFunction, Harness
from xlm.datamodule import Tokenizer


class MLMLoss(LossFunction[MLMBatch, MLMLossDict]):
    def __init__(
        self,
        loss_on_padding: bool = False,
        loss_on_visible_tokens: bool = False,
        model: Optional[MLMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.loss_on_padding = loss_on_padding
        self.loss_on_visible_tokens = loss_on_visible_tokens
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id_tensor = None
        self.flash_attn = None

    def configure(self, pl_module: Harness):
        self.mask_token_id_tensor = torch.tensor(  # type: ignore
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )

    def __call__(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    def loss_fn(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        if self.flash_attn is None:
            self.flash_attn = getattr(self.model, "force_flash_attn", False)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        targets = batch["target_ids"]
        assert targets is not None
        positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
        positions *= attention_mask  # technically not needed

        model = cast(MLMModel, self.model)
        logits = model(input_ids, attention_mask if not self.flash_attn else None, positions)

        ignore = torch.zeros_like(input_ids, dtype=torch.bool)
        if not self.loss_on_visible_tokens:
            ignore = ignore.logical_or(input_ids != self.mask_token_id_tensor)  # type: ignore
        targets[ignore] = -100
        if ignore.all():
            # Need to do this manually because pytorch doesn't do the logical thing. See https://github.com/pytorch/pytorch/issues/70348
            return {
                "loss": torch.tensor(
                    0.0,
                    device=logits.device,
                    dtype=logits.dtype,
                    requires_grad=True,
                )
            }

        logits_T = logits.transpose(1, 2)

        ce = torch.nn.functional.cross_entropy(
            logits_T, targets, reduction="mean", ignore_index=-100
        )
        # we can compute nlls by indexing using non-ignored tokens, but right now we will just use reduction=mean
        return {"loss": ce}
