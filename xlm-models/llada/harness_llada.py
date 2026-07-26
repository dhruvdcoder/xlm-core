"""FSDP-aware Harness for LLaDA training (global grad-norm clip + log)."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer

from xlm.harness import Harness


class FSDPHarness(Harness):
    """Harness with correct grad clipping / grad-norm logging under FSDP.

    Lightning's built-in ``gradient_clip_algorithm='norm'`` routes through
    ``FSDPPrecision`` and raises ``MisconfigurationException``. Base
    ``Harness.on_before_optimizer_step`` also logs only the *local shard*
    norm. See ``docs/guide/llms.md``.
    """

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        if gradient_clip_val is None:
            return
        root = self.trainer.strategy.model
        if isinstance(root, FSDP):
            root.clip_grad_norm_(max_norm=float(gradient_clip_val), norm_type=2.0)
            return
        return super().configure_gradient_clipping(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        root = self.trainer.strategy.model
        if not isinstance(root, FSDP):
            return super().on_before_optimizer_step(optimizer)

        local_sq = torch.zeros((), device=self.device, dtype=torch.float32)
        for p in root.parameters():
            if p.grad is not None:
                local_sq += p.grad.detach().float().pow(2).sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        global_norm = local_sq.sqrt()
        self.log(
            "Total gradient (norm)",
            global_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
            rank_zero_only=True,
            logger=True,
            add_dataloader_idx=False,
        )
