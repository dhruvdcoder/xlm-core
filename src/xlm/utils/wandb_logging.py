from __future__ import annotations

from typing import Optional

import lightning as L
import torch.distributed as dist
import wandb


class LogGradientsToWandb(L.pytorch.Callback):
    """Log top-k per-parameter gradient norms to W&B.

    Gradients are read in ``on_after_backward`` (before the optimizer step), then
    logged from ``on_train_batch_end`` so gradients match the completed backward.
    """

    def __init__(
        self,
        log_every_n_steps: int = 10,
        *,
        top_k: int = 100,
        table_caption_template: str = "Top-{k} gradient norms (global_step={step})",
        wandb_log_key_format: Optional[str] = "{tag}/gstep_{step:07d}",
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.top_k = top_k
        self.tag = "grads/topk_all_ranks_local_shard_norm"
        self.table_caption_template = table_caption_template
        self.wandb_log_key_format = wandb_log_key_format
        self._pending_grad_top: Optional[list[tuple[str, float]]] = None

    def on_after_backward(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        step = trainer.global_step
        if step % self.log_every_n_steps != 0:
            return

        # Collect local gradient norms for this rank.
        grads: list[tuple[str, float]] = []
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        for top_name, module in pl_module.top_level_named_modules():
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if param.grad is None:
                    continue
                g = param.grad.detach()
                # Use L2 norm to rank gradients; compute on device, then move scalar to CPU.
                grad_norm = float(g.float().norm().item())
                grads.append((f"rank{rank}/{top_name}/{name}", grad_norm))

        # Gather local grads from all ranks so top-k is computed globally.
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered: list[list[tuple[str, float]] | None] = [None] * world_size
            dist.all_gather_object(gathered, grads)
            all_grads = [item for per_rank in gathered if per_rank for item in per_rank]
        else:
            all_grads = grads

        if not all_grads:
            self._pending_grad_top = None
            return

        # Log once from rank 0 to avoid duplicate W&B entries.
        if dist.is_available() and dist.is_initialized() and rank != 0:
            return

        all_grads.sort(key=lambda x: x[1], reverse=True)
        top = all_grads[: self.top_k]
        # Last backward in a grad-accum window wins (accumulated grads on params).
        self._pending_grad_top = top

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs=None,
        batch=None,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> None:
        if self._pending_grad_top is None:
            return
        top = self._pending_grad_top
        self._pending_grad_top = None

        exp = getattr(pl_module.logger, "experiment", None)
        if exp is None:
            return

        step = int(trainer.global_step)
        table_caption = self.table_caption_template.format(k=len(top), step=step)
        table = wandb.Table(columns=["param", "grad_norm"])
        for n, v in top:
            table.add_data(n, v)

        if self.wandb_log_key_format is None:
            log_key = self.tag
        else:
            log_key = self.wandb_log_key_format.format(
                tag=self.tag,
                step=step,
                k=len(top),
            )
        payload = {
            log_key: wandb.plot.bar(
                table,
                "param",
                "grad_norm",
                title=table_caption,
            ),
            "grads/logged_at_global_step": step,
        }
        # Omit step=...: W&B requires monotonic steps vs other Lightning logs; passing
        # trainer.global_step here still races validation / other hooks.
        exp.log(payload)
