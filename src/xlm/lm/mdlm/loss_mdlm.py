from typing import Optional, cast

import torch
from .types_mdlm import MDLMBatch, MDLMLossDict, MDLMModel
from xlm.harness import LossFunction, Harness
from xlm.datamodule import Tokenizer


class MDLMLoss(LossFunction[MDLMBatch, MDLMLossDict]):
    def __init__(
        self,
        loss_on_padding: bool = False,
        loss_on_visible_tokens: bool = False,
        model: Optional[MDLMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.loss_on_padding = loss_on_padding
        self.loss_on_visible_tokens = loss_on_visible_tokens
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id_tensor = None

    def configure(self, pl_module: Harness):
        self.mask_token_id_tensor = torch.tensor(  # type: ignore
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )

    def __call__(
        self,
        batch: MDLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MDLMLossDict:
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    def loss_fn(
        self,
        batch: MDLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MDLMLossDict:

        # fmt: off
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        targets = batch["target_ids"]
        noise_rate = batch["noise_rate"]
        total_noise = batch["total_noise"]
        t = batch["t"]
        # fmt: on
        assert targets is not None
        positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
        positions *= attention_mask  # technically not needed

        model = cast(MDLMModel, self.model)
        logits = model(input_ids, total_noise, attention_mask, positions)

        ignore = torch.zeros_like(input_ids, dtype=torch.bool)
        if not self.loss_on_visible_tokens:
            ignore = ignore.logical_or(input_ids != self.mask_token_id_tensor)  # type: ignore
        targets[ignore] = -100

        logits_T = logits.transpose(1, 2)

        ce = torch.nn.functional.cross_entropy(
            logits_T, targets, reduction="none", ignore_index=-100
        )  # (batch, seq_len)
        weight = noise_rate / torch.expm1(total_noise)  #
        kl = ce * weight.unsqueeze(-1)  # (batch, seq_len)
        not_ignore = ~ignore
        if ignore.all():
            # Need to do this manually because pytorch doesn't do the logical thing. See https://github.com/pytorch/pytorch/issues/70348
            loss = torch.tensor(
                0.0,
                device=logits.device,
                dtype=logits.dtype,
                requires_grad=True,
            )
        else:
            loss = kl[not_ignore].mean()
        nlls = kl[not_ignore]
        # we can compute nlls by indexing using non-ignored tokens, but right now we will just use reduction=mean
        return {"loss": loss, "nlls": nlls}
