"""Subprocess entrypoint for the CPU (gloo) FSDP optimizer-coverage test.

Launched by tests.integration._runner.run_cpu_distributed under
torch.distributed.run. Each rank:

1. Initialises a gloo process group.
2. Builds a bare xlm.harness.Harness around a tiny decoder-like model
   (Embedding -> 2 x Block -> LayerNorm -> Linear head) and wraps the
   *harness* in FullyShardedDataParallel with a per-Block auto-wrap policy,
   exactly the way Lightning's FSDPStrategy does. The embedding, final norm
   and head therefore land in the root FSDP unit.
3. Builds two optimizers: the legacy one from harness.model.parameters()
   and the fixed one from Harness.optimizer_named_parameters(). For each it
   runs Harness.check_optimizer_covers_trainable_params and takes one SGD
   step, recording whether the root-unit weights actually moved.
4. Exercises FSDPHarness.configure_gradient_clipping on the root.
5. Writes rank_<RANK>.json into --result-dir.

Config schema (JSON, passed via --config-json)::

    {"use_orig_params": false}
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--config-json", type=str, default="{}")
    return parser.parse_args()


def _build_tiny_model():
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.lin = nn.Linear(d, d)
            self.norm = nn.LayerNorm(d)

        def forward(self, x):
            return self.norm(self.lin(x))

    class Tiny(nn.Module):
        def __init__(self, vocab: int = 16, d: int = 8):
            super().__init__()
            self.embed = nn.Embedding(vocab, d)
            self.layers = nn.ModuleList([Block(d) for _ in range(2)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, vocab)

        def forward(self, ids):
            x = self.embed(ids)
            for layer in self.layers:
                x = layer(x)
            return self.head(self.norm(x))

    return Tiny(), Block


def _bare_harness(harness_cls):
    """Instantiate a Harness without the Hydra / datamodule machinery.

    Harness has no forward of its own (training goes through compute_loss),
    but the FSDP root calls forward on the wrapped module, so we add one that
    delegates to the model.
    """
    import lightning as L
    from omegaconf import OmegaConf

    class _Forwarding(harness_cls):  # type: ignore[misc,valid-type]
        def forward(self, ids):  # type: ignore[override]
            return self.model(ids)

    harness = _Forwarding.__new__(_Forwarding)
    L.LightningModule.__init__(harness)
    model, block_cls = _build_tiny_model()
    harness.model = model
    harness.config = OmegaConf.create({"lightning_module": {}})
    return harness, block_cls


def _attach_fake_trainer(harness, root) -> None:
    from types import SimpleNamespace

    harness._trainer = SimpleNamespace(strategy=SimpleNamespace(model=root))


def _root_unit_checksums(root) -> Dict[str, float]:
    """Checksums of every parameter owned by the root unit (not in a block)."""
    out: Dict[str, float] = {}
    for name, p in root.named_parameters():
        if ".layers." in name:
            continue
        out[name] = float(p.detach().float().sum().item())
    return out


def _numel(params) -> int:
    return int(sum(p.numel() for p in params))


def _try_check(harness, optimizer) -> Tuple[bool, str]:
    try:
        harness.check_optimizer_covers_trainable_params(optimizer)
        return False, ""
    except RuntimeError as exc:  # expected failure mode
        return True, str(exc)[:400]


def _step_and_report_root_motion(root, optimizer, ids, torch) -> bool:
    before = _root_unit_checksums(root)
    optimizer.zero_grad(set_to_none=True)
    out = root(ids)
    out.float().pow(2).mean().backward()
    optimizer.step()
    after = _root_unit_checksums(root)
    return any(abs(before[k] - after[k]) > 0 for k in before)


def _run_one_rank(rank: int, world_size: int, config: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    from xlm.harness import FSDPHarness, Harness

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", init_method="env://", rank=rank, world_size=world_size
        )
    torch.manual_seed(0)
    use_orig_params = bool(config.get("use_orig_params", False))
    cpu = torch.device("cpu")

    result: Dict[str, Any] = {
        "rank": rank,
        "world_size": world_size,
        "use_orig_params": use_orig_params,
    }

    # ---- legacy optimizer source: harness.model.parameters() -------------
    harness, block_cls = _bare_harness(Harness)
    root = FSDP(
        harness,
        auto_wrap_policy=ModuleWrapPolicy({block_cls}),
        use_orig_params=use_orig_params,
        device_id=cpu,
    )
    _attach_fake_trainer(harness, root)
    assert harness.fsdp_root() is root
    trainable_numel = _numel(p for p in root.parameters() if p.requires_grad)
    result["trainable_numel"] = trainable_numel

    legacy_params = [p for p in harness.model.parameters() if p.requires_grad]
    legacy_opt = torch.optim.SGD(legacy_params, lr=0.1)
    raised, msg = _try_check(harness, legacy_opt)
    ids = torch.randint(0, 16, (2, 4))
    result["legacy"] = {
        "optimizer_numel": _numel(legacy_params),
        "raised": raised,
        "error_snippet": msg,
        "root_moved": _step_and_report_root_motion(root, legacy_opt, ids, torch),
    }

    # ---- fixed optimizer source: Harness.optimizer_named_parameters() ----
    harness2, _ = _bare_harness(Harness)
    root2 = FSDP(
        harness2,
        auto_wrap_policy=ModuleWrapPolicy({block_cls}),
        use_orig_params=use_orig_params,
        device_id=cpu,
    )
    _attach_fake_trainer(harness2, root2)
    named = harness2.optimizer_named_parameters()
    fixed_params = [p for _, p in named if p.requires_grad]
    fixed_opt = torch.optim.SGD(fixed_params, lr=0.1)
    raised2, msg2 = _try_check(harness2, fixed_opt)
    result["fixed"] = {
        "optimizer_numel": _numel(fixed_params),
        "raised": raised2,
        "error_snippet": msg2,
        "names": [n for n, _ in named],
        "root_moved": _step_and_report_root_motion(root2, fixed_opt, ids, torch),
    }

    # ---- FSDPHarness.configure_gradient_clipping on the root -------------
    fsdp_harness, _ = _bare_harness(FSDPHarness)
    root3 = FSDP(
        fsdp_harness,
        auto_wrap_policy=ModuleWrapPolicy({block_cls}),
        use_orig_params=use_orig_params,
        device_id=cpu,
    )
    _attach_fake_trainer(fsdp_harness, root3)
    opt3 = torch.optim.SGD(
        [p for _, p in fsdp_harness.optimizer_named_parameters()], lr=0.1
    )
    opt3.zero_grad(set_to_none=True)
    (root3(ids).float().pow(2).mean() * 1e3).backward()

    def _global_grad_norm() -> float:
        sq = torch.zeros((), dtype=torch.float32)
        for p in root3.parameters():
            if p.grad is not None:
                sq += p.grad.detach().float().pow(2).sum()
        dist.all_reduce(sq, op=dist.ReduceOp.SUM)
        return float(sq.sqrt().item())

    norm_before = _global_grad_norm()
    fsdp_harness.configure_gradient_clipping(opt3, 0.5, "norm")
    norm_after = _global_grad_norm()
    result["clip"] = {"norm_before": norm_before, "norm_after": norm_after}

    dist.barrier()
    dist.destroy_process_group()
    return result


def main() -> None:
    args = _parse_args()
    config = json.loads(args.config_json)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    payload: Dict[str, Any]
    try:
        payload = _run_one_rank(rank, world_size, config)
        payload["ok"] = True
        payload["error"] = None
    except Exception:  # noqa: BLE001 - serialised for the parent test
        payload = {
            "rank": rank,
            "world_size": world_size,
            "ok": False,
            "error": traceback.format_exc(),
        }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    with (args.result_dir / f"rank_{rank}.json").open("w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
