"""Diagnostics callback for FSDP runs.

Reports, on rank 0:
    1. Resolved FSDP strategy settings (sharding, mixed precision, policies).
    2. Post-wrap module-tree statistics (FSDP units, CheckpointWrappers, top names).
    3. Per-phase peak GPU memory (forward / backward / optimizer) for the first
       ``num_logged_batches`` training steps.

These three signals together let us distinguish OOM root causes:
    - if (1) shows the wrong policy / dtype, the config is not right, eg. YAML did not merge correctly, etc.;
    - if (2) shows only one FSDP unit / no CheckpointWrappers, the auto-wrap or
      activation-checkpointing policy did not fire on the target layer class
      (so the model is effectively un-sharded or fully materialized);
    - if (3) shows the peak in forward, activation memory dominates; if in
      optimizer, parameter / state shards dominate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from lightning import Callback, LightningModule, Trainer
from lightning_utilities.core.rank_zero import rank_zero_only

from xlm.utils.rank_zero import RankedLogger


def _try_import_fsdp_class() -> Optional[Type[Any]]:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel

        return FullyShardedDataParallel
    except Exception:
        return None


def _try_import_checkpoint_wrapper_class() -> Optional[Type[Any]]:
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        return CheckpointWrapper
    except Exception:
        return None


def _format_class_set(value: Any) -> str:
    """Render an auto-wrap / activation-checkpointing policy in a stable way."""
    if value is None:
        return "None"
    if isinstance(value, (set, frozenset, list, tuple)):
        try:
            names = sorted(getattr(c, "__name__", repr(c)) for c in value)
            return "{" + ", ".join(names) + "}"
        except Exception:
            return repr(value)
    return repr(value)


def _module_classname(m: Any) -> str:
    return type(m).__name__


def _count_and_sample(
    pl_module: LightningModule,
    target_cls: Optional[Type[Any]],
    top_k: int,
) -> Tuple[int, List[str]]:
    if target_cls is None:
        return 0, []
    names: List[str] = []
    count = 0
    for name, mod in pl_module.named_modules():
        if isinstance(mod, target_cls):
            count += 1
            if len(names) < top_k:
                names.append(name or "<root>")
    return count, names


class FSDPDiagnosticsCallback(Callback):
    """Lightning Callback that surfaces FSDP wrap and per-phase memory stats."""

    def __init__(
        self,
        num_logged_batches: int = 3,
        log_module_tree_top_k: int = 5,
        log_to_logger: bool = True,
    ) -> None:
        """
        Args:
            num_logged_batches: How many of the first training batches to
                instrument with peak-memory measurements. Set to 0 to disable
                per-batch logging.
            log_module_tree_top_k: How many sample module names of each kind
                (FSDP unit, CheckpointWrapper) to print.
            log_to_logger: If True and a Trainer logger is configured, also
                push memory metrics through ``trainer.logger.log_metrics``.
        """
        super().__init__()
        self.num_logged_batches = int(num_logged_batches)
        self.log_module_tree_top_k = int(log_module_tree_top_k)
        self.log_to_logger = bool(log_to_logger)
        self._batches_seen = 0
        self._log = RankedLogger(name=__name__, rank_zero_only=False)

    # ------------------------------------------------------------------
    # Strategy / config dump (runs before FSDP wrap, so only inspects config)
    # ------------------------------------------------------------------
    @rank_zero_only
    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        strategy = trainer.strategy
        info: Dict[str, Any] = {
            "stage": stage,
            "strategy_class": _module_classname(strategy),
            "trainer_precision": str(getattr(trainer, "precision", "?")),
            "world_size": getattr(trainer, "world_size", "?"),
            "num_nodes": getattr(trainer, "num_nodes", "?"),
            "devices_per_node": getattr(trainer, "num_devices", "?"),
        }
        for attr in (
            "sharding_strategy",
            "cpu_offload",
            "use_orig_params",
            "auto_wrap_policy",
            "activation_checkpointing_policy",
            "mixed_precision",
            "state_dict_type",
        ):
            if hasattr(strategy, attr):
                value = getattr(strategy, attr)
                if attr in (
                    "auto_wrap_policy",
                    "activation_checkpointing_policy",
                ):
                    info[attr] = _format_class_set(value)
                elif attr == "mixed_precision":
                    info[attr] = self._render_mixed_precision(value)
                else:
                    info[attr] = repr(value)

        precision_plugin = getattr(strategy, "precision_plugin", None)
        if precision_plugin is not None:
            info["precision_plugin"] = _module_classname(precision_plugin)
            mp_cfg = getattr(precision_plugin, "mixed_precision_config", None)
            if mp_cfg is not None:
                info["precision_plugin.mixed_precision_config"] = (
                    self._render_mixed_precision(mp_cfg)
                )

        self._log.info("[FSDPDiagnostics setup] " + self._kv(info), rank=0)

    # ------------------------------------------------------------------
    # Post-wrap structure dump
    # ------------------------------------------------------------------
    def on_fit_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        fsdp_cls = _try_import_fsdp_class()
        ckpt_cls = _try_import_checkpoint_wrapper_class()

        fsdp_count, fsdp_names = _count_and_sample(
            pl_module, fsdp_cls, self.log_module_tree_top_k
        )
        ckpt_count, ckpt_names = _count_and_sample(
            pl_module, ckpt_cls, self.log_module_tree_top_k
        )

        local_params = sum(
            p.numel() for p in pl_module.parameters() if p.requires_grad
        )
        local_param_bytes = sum(
            p.numel() * p.element_size()
            for p in pl_module.parameters()
            if p.requires_grad
        )

        # Local view; with FULL_SHARD this is the per-rank shard.
        info: Dict[str, Any] = {
            "fsdp_units": fsdp_count,
            "fsdp_unit_sample": fsdp_names,
            "checkpoint_wrappers": ckpt_count,
            "checkpoint_wrapper_sample": ckpt_names,
            "local_trainable_params": local_params,
            "local_trainable_param_MiB": round(
                local_param_bytes / (1024**2), 2
            ),
        }
        # Each rank logs its own line so we can compare shard sizes.
        self._log.info("[FSDPDiagnostics on_fit_start] " + self._kv(info))

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # Per-phase peak memory for the first N batches
    # ------------------------------------------------------------------
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self._should_log_batch():
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._log_mem(
            tag="batch_start",
            batch_idx=batch_idx,
            trainer=trainer,
        )

    def on_after_backward(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not self._should_log_batch():
            return
        self._log_mem(
            tag="after_backward",
            batch_idx=trainer.global_step,
            trainer=trainer,
        )

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if not self._should_log_batch():
            return
        self._log_mem(
            tag="before_optimizer_step",
            batch_idx=trainer.global_step,
            trainer=trainer,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self._should_log_batch():
            return
        self._log_mem(
            tag="batch_end",
            batch_idx=batch_idx,
            trainer=trainer,
        )
        self._batches_seen += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _should_log_batch(self) -> bool:
        return (
            self.num_logged_batches > 0
            and self._batches_seen < self.num_logged_batches
            and torch.cuda.is_available()
        )

    def _log_mem(
        self,
        *,
        tag: str,
        batch_idx: int,
        trainer: Trainer,
    ) -> None:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        msg = {
            "phase": tag,
            "batch": batch_idx,
            "allocated_GiB": round(allocated, 3),
            "reserved_GiB": round(reserved, 3),
            "peak_allocated_GiB": round(peak, 3),
        }
        self._log.info("[FSDPDiagnostics mem] " + self._kv(msg))
        if self.log_to_logger and trainer.logger is not None:
            try:
                trainer.logger.log_metrics(
                    {
                        f"fsdp_diag/{tag}/allocated_GiB": allocated,
                        f"fsdp_diag/{tag}/reserved_GiB": reserved,
                        f"fsdp_diag/{tag}/peak_GiB": peak,
                    },
                    step=trainer.global_step,
                )
            except Exception:
                # Logger may not be ready (e.g. before fit fully starts).
                pass

    @staticmethod
    def _render_mixed_precision(value: Any) -> str:
        if value is None:
            return "None"
        for attr in (
            "param_dtype",
            "reduce_dtype",
            "buffer_dtype",
            "cast_forward_inputs",
            "keep_low_precision_grads",
        ):
            if hasattr(value, attr):
                # Looks like a torch MixedPrecision dataclass; build a compact summary.
                pieces: List[str] = []
                for a in (
                    "param_dtype",
                    "reduce_dtype",
                    "buffer_dtype",
                    "cast_forward_inputs",
                    "keep_low_precision_grads",
                ):
                    if hasattr(value, a):
                        pieces.append(f"{a}={getattr(value, a)!r}")
                return "MixedPrecision(" + ", ".join(pieces) + ")"
        return repr(value)

    @staticmethod
    def _kv(info: Dict[str, Any]) -> str:
        return " ".join(f"{k}={v}" for k, v in info.items())
