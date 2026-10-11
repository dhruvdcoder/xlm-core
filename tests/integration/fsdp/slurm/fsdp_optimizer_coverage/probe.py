"""Lightning callback that records FSDP optimizer-coverage evidence per rank.

Injected into a real xlm training run by script.py via
+callbacks.fsdp_probe._target_=...FSDPCoverageProbe. It never asserts; it
only observes and stores a dict in the module-level RESULT so that script.py
can serialise it next to any exception the run raised.

What it records
===============

on_fit_start (after FSDPStrategy wrapped the module and built optimizers):
    * is_fsdp, use_orig_params, harness class
    * trainable_numel over the FSDP root vs optimizer_numel over all groups
    * coverage_ok: whether Harness.check_optimizer_covers_trainable_params
      passes on the live optimizer
    * root_checksums_before: per-parameter checksums of the local shard of
      every parameter owned by the root unit (outside the wrapped blocks)
on_before_optimizer_step (every step):
    * local_grad_norm (this rank's shards only) and global_grad_norm
      (all-reduced), to compare with what the harness logs
on_train_batch_end (every step):
    * logged_total_grad_norm: trainer.callback_metrics["Total gradient (norm)"]
on_train_end:
    * root_checksums_after and root_moved
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from lightning import Callback, LightningModule, Trainer

RESULT: Dict[str, Any] = {}


def _try_fsdp_class():
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel

        return FullyShardedDataParallel
    except Exception:  # pragma: no cover
        return None


def _root_unit_checksums(root: torch.nn.Module, block_marker: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, p in root.named_parameters():
        if block_marker in name:
            continue
        out[name] = float(p.detach().float().sum().item())
    return out


class FSDPCoverageProbe(Callback):
    def __init__(self, block_marker: str = ".encoder.") -> None:
        super().__init__()
        self.block_marker = block_marker
        self._local_norms: list = []
        self._global_norms: list = []
        self._logged_norms: list = []

    # ------------------------------------------------------------------
    def _root(self, trainer: Trainer) -> Optional[torch.nn.Module]:
        FSDP = _try_fsdp_class()
        root = getattr(trainer.strategy, "model", None)
        if FSDP is not None and isinstance(root, FSDP):
            return root
        return None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        root = self._root(trainer)
        RESULT["harness_class"] = type(pl_module).__name__
        RESULT["is_fsdp"] = root is not None
        RESULT["use_orig_params"] = (
            bool(getattr(trainer.strategy, "kwargs", {}).get("use_orig_params"))
            if root is not None
            else None
        )
        source = root if root is not None else pl_module
        RESULT["trainable_numel"] = int(
            sum(p.numel() for p in source.parameters() if p.requires_grad)
        )
        optimizers = trainer.optimizers
        RESULT["num_optimizers"] = len(optimizers)
        seen: set = set()
        optimizer_numel = 0
        for opt in optimizers:
            for group in opt.param_groups:
                for p in group["params"]:
                    if id(p) in seen:
                        continue
                    seen.add(id(p))
                    optimizer_numel += p.numel()
        RESULT["optimizer_numel"] = int(optimizer_numel)
        coverage_ok = True
        coverage_error = ""
        check = getattr(pl_module, "check_optimizer_covers_trainable_params", None)
        if check is not None:
            for opt in optimizers:
                try:
                    check(opt)
                except RuntimeError as exc:
                    coverage_ok = False
                    coverage_error = str(exc)[:400]
        RESULT["coverage_ok"] = coverage_ok
        RESULT["coverage_error"] = coverage_error
        if root is not None:
            RESULT["root_checksums_before"] = _root_unit_checksums(
                root, self.block_marker
            )
            RESULT["num_root_unit_params"] = len(RESULT["root_checksums_before"])

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer
    ) -> None:
        root = self._root(trainer)
        if root is None:
            return
        import torch.distributed as dist

        sq = torch.zeros((), device=pl_module.device, dtype=torch.float32)
        for p in root.parameters():
            if p.grad is not None:
                sq += p.grad.detach().float().pow(2).sum()
        local = float(sq.sqrt().item())
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sq, op=dist.ReduceOp.SUM)
        self._local_norms.append(local)
        self._global_norms.append(float(sq.sqrt().item()))

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
    ) -> None:
        value = trainer.callback_metrics.get("Total gradient (norm)")
        if value is not None:
            self._logged_norms.append(float(value))

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        root = self._root(trainer)
        RESULT["global_step"] = int(trainer.global_step)
        RESULT["local_grad_norms"] = self._local_norms
        RESULT["global_grad_norms"] = self._global_norms
        RESULT["logged_total_grad_norms"] = self._logged_norms
        if root is None:
            return
        after = _root_unit_checksums(root, self.block_marker)
        before = RESULT.get("root_checksums_before", {})
        RESULT["root_checksums_after"] = after
        RESULT["root_moved"] = any(
            abs(before.get(k, 0.0) - after[k]) > 0 for k in after
        )
