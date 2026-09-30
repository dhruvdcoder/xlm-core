"""FRM loss: token CE on the clean-prediction interpolant, with self-cond / FPF."""

from typing import Optional, cast

import torch
import torch.nn.functional as F

from xlm.datamodule import Tokenizer
from xlm.harness import Harness, LossFunction
from xlm.utils.nn import masked_mean

from .flow_frm import apply_clamp, interpolant, one_hot_tokens
from .types_frm import FRMBatch, FRMLossDict, FRMModel


class FRMLoss(LossFunction[FRMBatch, FRMLossDict]):
    def __init__(
        self,
        loss_on_padding: bool = True,
        loss_on_clamped: bool = True,
        self_cond_prob: float = 0.5,
        fpf_prob: float = 0.0,
        fpf_rollout_depth: int = 16,
        t_eps: float = 1e-3,
        model: Optional[FRMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.loss_on_padding = loss_on_padding
        self.loss_on_clamped = loss_on_clamped
        self.self_cond_prob = self_cond_prob
        self.fpf_prob = fpf_prob
        self.fpf_rollout_depth = fpf_rollout_depth
        self.t_eps = t_eps
        self.model = model
        self.tokenizer = tokenizer

    def configure(self, pl_module: Harness) -> None:
        return

    def __call__(
        self,
        batch: FRMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FRMLossDict:
        return self.loss_fn(batch, batch_idx, dataloader_idx, dataloader_name)

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device) * (
            1.0 - 2 * self.t_eps
        ) + self.t_eps

    def _fpf_carry(
        self,
        model: FRMModel,
        x1: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
        clue_ids: Optional[torch.Tensor],
        clamp_mask: Optional[torch.Tensor],
        clue_onehot: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Stopgrad Euler rollout from ``t_start ~ U(0, t)`` to ``t``."""
        batch = t.shape[0]
        t_start = torch.rand(batch, device=t.device) * t
        depth = max(int(self.fpf_rollout_depth), 1)
        dt = (t - t_start) / depth
        noise = torch.randn_like(x1)
        x = interpolant(x1, t_start, noise)
        x = apply_clamp(x, clue_onehot, clamp_mask) if clue_onehot is not None else x
        s = None
        with torch.no_grad():
            for step in range(depth):
                t_now = t_start + dt * step
                logits = model(
                    x,
                    t_now,
                    attention_mask=attention_mask,
                    positions=positions,
                    s=s,
                    clue_ids=clue_ids,
                    clamp_mask=clamp_mask,
                )
                s = F.softmax(logits, dim=-1)
                denom = (1.0 - t_now).clamp(min=1e-5).view(-1, 1, 1)
                x = x + (s - x) / denom * dt.view(-1, 1, 1)
                if clue_onehot is not None:
                    x = apply_clamp(x, clue_onehot, clamp_mask)
        assert s is not None
        return s.detach()

    def _self_cond_carry(
        self,
        model: FRMModel,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
        clue_ids: Optional[torch.Tensor],
        clamp_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = model(
                x_t,
                t,
                attention_mask=attention_mask,
                positions=positions,
                s=None,
                clue_ids=clue_ids,
                clamp_mask=clamp_mask,
            )
            s = F.softmax(logits, dim=-1)
        return s.detach()

    def loss_fn(
        self,
        batch: FRMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FRMLossDict:
        model = cast(FRMModel, self.model)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        targets = batch["target_ids"]
        assert targets is not None
        clamp_mask = batch.get("clamp_mask")
        if clamp_mask is not None:
            clamp_mask = clamp_mask.to(dtype=torch.bool)
        clue_ids = batch.get("clue_ids")
        vocab_size = model.num_embeddings
        positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)

        x1 = one_hot_tokens(input_ids, vocab_size)
        clue_onehot = (
            one_hot_tokens(clue_ids, vocab_size) if clue_ids is not None else None
        )
        if clue_onehot is not None:
            x1 = apply_clamp(x1, clue_onehot, clamp_mask)

        t = self._sample_t(input_ids.shape[0], input_ids.device)
        x_t = interpolant(x1, t)
        if clue_onehot is not None:
            x_t = apply_clamp(x_t, clue_onehot, clamp_mask)

        s = None
        training = bool(getattr(model, "training", True))
        if training:
            use_fpf = self.fpf_prob > 0 and (
                torch.rand((), device=t.device) < self.fpf_prob
            )
            use_self_cond = (not use_fpf) and self.self_cond_prob > 0 and (
                torch.rand((), device=t.device) < self.self_cond_prob
            )
            if use_fpf:
                s = self._fpf_carry(
                    model,
                    x1,
                    t,
                    attention_mask,
                    positions,
                    clue_ids,
                    clamp_mask,
                    clue_onehot,
                )
            elif use_self_cond:
                s = self._self_cond_carry(
                    model,
                    x_t,
                    t,
                    attention_mask,
                    positions,
                    clue_ids,
                    clamp_mask,
                )

        logits = model(
            x_t,
            t,
            attention_mask=attention_mask,
            positions=positions,
            s=s,
            clue_ids=clue_ids,
            clamp_mask=clamp_mask,
        )

        ce_targets = targets.clone()
        ignore = torch.zeros_like(ce_targets, dtype=torch.bool)
        if not self.loss_on_padding:
            ignore = ignore | ~attention_mask
        if not self.loss_on_clamped and clamp_mask is not None:
            ignore = ignore | clamp_mask
        ce_targets = ce_targets.masked_fill(ignore, -100)

        token_loss = F.cross_entropy(
            logits.transpose(1, 2),
            ce_targets,
            reduction="none",
            ignore_index=-100,
        )
        loss = masked_mean(token_loss.flatten(), ~ignore.flatten(), dim=-1)
        return FRMLossDict(loss=loss)
