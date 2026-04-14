from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
import torch
from torch import nn


class DreamOnLoss:

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        token_reweighting: bool = False,
        alpha: float = 0.25,
        gamma: float = 2.0,
        time_reweighting: Optional[Literal["linear"]] = None,
        weight_eos: bool = False,
        max_delete: int = 64,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.token_reweighting = token_reweighting
        self.alpha = alpha
        self.gamma = gamma
        self.time_reweighting = time_reweighting
        self.weight_eos = weight_eos
        self.max_delete = max_delete

    def __call__(
        self,
        batch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ):
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    def loss_fn(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        del batch_idx, dataloader_idx, dataloader_name
        input_ids = batch["input_ids"].to(self.model.device)
        labels = batch["labels"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)
        position_ids = batch["position_ids"].to(self.model.device)
        loss_mask = batch["loss_mask"].to(self.model.device)
        t = batch["t"].to(self.model.device)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        if attention_mask.dim() == 2:
        # Input is (B, S) -> need to create pairwise mask (B, S, S)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),  # (B, 1, S, 1)
                attention_mask.unsqueeze(1).unsqueeze(-1)   # (B, 1, S, 1)
            )  # Result: (B, 1, S, S)

        elif attention_mask.dim() == 3:
        # Already (B, S, S), just add head dimension
            attention_mask = attention_mask.unsqueeze(1)  # (B, 1, S, S)
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

        loss_mask = loss_mask.reshape(-1)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = output.logits

        shift_logits = torch.cat(
            [logits[:, 0:1], logits[:, :-1]], dim=1
        ).contiguous()
        shift_labels = labels.contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        # We use weighted loss
        loss_mask = loss_mask.to(loss.device)
        loss = loss.masked_fill(~loss_mask, 0)
        if self.config.diffusion.token_reweighting:
            loss = (
                self.config.diffusion.alpha
                * (1 - torch.exp(-loss)) ** self.config.diffusion.gamma
                * loss
            )

        if self.config.diffusion.time_reweighting == "original":
            raise NotImplementedError
            weight = 1 / t[:, None].float().expand(labels.size())
        elif self.config.diffusion.time_reweighting == "linear":
            weight = 1 - t.float().expand(labels.size())
        else:
            raise NotImplementedError

        loss = loss * weight.reshape(-1)

        if self.config.diffusion.weight_eos and self.config.data.max_delete > 0:
            non_eos_mask = (shift_labels != self.tokenizer.eos_token_id) & loss_mask
            non_eos_loss = loss.clone()  
            non_eos_loss[~non_eos_mask] = 0  
            non_eos_count = non_eos_mask.sum().item() 
            non_eos_loss = non_eos_loss.sum()  

            
            eos_mask = (shift_labels == self.tokenizer.eos_token_id) & loss_mask
            eos_loss = loss.clone()  
            eos_loss[~eos_mask] = 0  
            eos_count = eos_mask.sum().item()  
            eos_loss = eos_loss.sum() / eos_count  

            
            loss = (non_eos_loss + eos_loss) / (non_eos_count + 1)  
        else:
            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = (
                    torch.distributed.get_world_size()
                )
            else:
                dp_size = 1

            loss = torch.sum(loss) / valid_token_this_rank * dp_size

        return {"loss": loss}
