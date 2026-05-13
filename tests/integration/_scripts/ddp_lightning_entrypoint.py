"""Subprocess entrypoint for the Lightning-Trainer DDP integration test.

Wraps a single :class:`xlm.datamodule.DatasetManager` in a tiny
:class:`xlm.datamodule.TextDataModule`, attaches a no-op
:class:`lightning.LightningModule` whose only job is to record the
``id`` of every example each rank actually consumes, and runs a CPU
DDP :class:`lightning.Trainer` for ``max_epochs`` epochs.

Why a separate entrypoint
=========================

The matrix tests in
``test_dataset_manager_paths.py`` and the bespoke runner tests in
``test_dataset_manager_ddp_cpu.py`` already exhaustively cover
``DatasetManager`` in isolation.  This entrypoint exists for the
*tiered* Lightning case: it confirms that ``rank`` / ``world_size``
are correctly threaded from the Trainer through ``TextDataModule``
into ``DatasetManager`` -- a path that is otherwise impossible to
exercise without a real Trainer.

Same launch protocol as :mod:`tests.integration._scripts.ddp_dsm_entrypoint`:

* ``--result-dir <dir>`` -- per-rank ``rank_<RANK>.json`` written here.
* ``--config-json <json>`` -- forwarded ``DatasetManager`` kwargs and
  Trainer overrides.

Config schema
-------------

::

    {
      "dsm_kwargs": {...},          # DatasetManager(**...)
      "trainer_kwargs": {           # Trainer(**...)
        "max_epochs": 1,
        "limit_train_batches": 5
      }
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--config-json", type=str, default="{}")
    return parser.parse_args()


def _make_tokenizer():
    from xlm.datamodule import SimpleSpaceTokenizer

    return SimpleSpaceTokenizer(vocab=[str(i) for i in range(50)])


def _build_lightning_module(collected_ids: List[int]):
    """A no-op LightningModule that records every batch id it sees."""
    import lightning as L
    import torch
    import torch.nn as nn

    class _RecorderLM(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            # Trainer requires the model to have at least one parameter
            # to set up optimizers; the optimizer never gets a useful
            # gradient because we never use this parameter in the loss.
            self.dummy = nn.Parameter(torch.zeros(1))

        def training_step(self, batch, batch_idx):  # type: ignore[override]
            collected_ids.extend(int(x) for x in batch["ids"])
            # Loss must depend on a parameter or Lightning warns that
            # the optimizer has no gradient.
            return self.dummy.sum() * 0.0

        def configure_optimizers(self):  # type: ignore[override]
            return torch.optim.SGD(self.parameters(), lr=0.0)

    return _RecorderLM()


def _run_one_rank(result_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a Lightning Trainer with ``strategy="ddp"`` on CPU."""
    import lightning as L
    import torch

    from tests.datamodule_helpers import (
        EXAMPLE_TOKEN_LEN,
        IdTrackingCollator,
        build_inmem_datasets,
        patch_dataset_manager_download,
    )
    from xlm.datamodule import DatasetManager, TextDataModule

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    L.seed_everything(int((config.get("trainer_kwargs") or {}).get("seed", 0)))

    patch_dataset_manager_download(build_inmem_datasets())
    tokenizer = _make_tokenizer()
    collator = IdTrackingCollator(tokenizer=tokenizer)

    dsm_kwargs = dict(config.get("dsm_kwargs") or {})
    dsm_kwargs.setdefault("full_name", "mem/raw_large/train")
    dsm_kwargs.setdefault("full_name_debug", dsm_kwargs["full_name"])
    dsm_kwargs.setdefault("collator", collator)
    dsm_kwargs.setdefault(
        "preprocess_function",
        "tests.datamodule_helpers.example_to_input_ids",
    )
    dsm_kwargs.setdefault("columns_to_keep", ["id"])
    dsm_kwargs.setdefault(
        "dataloader_kwargs",
        {"batch_size": 4, "num_workers": 1, "pin_memory": False},
    )
    dsm_kwargs.setdefault("iterable_dataset_shards", 4)
    dsm_kwargs.setdefault("use_manual_cache", False)
    dsm_kwargs.setdefault("stages", ["fit"])

    train_dsm = DatasetManager(**dsm_kwargs)

    cache_dir = result_dir / f"_lightning_cache_rank_{rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dm = TextDataModule(
        manual_cache_dir=str(cache_dir),
        tokenizer=tokenizer,
        dataset_managers={"train": {"main": train_dsm}},
        block_size=EXAMPLE_TOKEN_LEN,
        global_batch_size=dsm_kwargs["dataloader_kwargs"]["batch_size"]
        * world_size,
    )

    collected_ids: List[int] = []
    model = _build_lightning_module(collected_ids)

    trainer_kwargs = dict(config.get("trainer_kwargs") or {})
    trainer_kwargs.pop("seed", None)
    trainer_kwargs.setdefault("max_epochs", 1)
    trainer_kwargs.setdefault("limit_train_batches", 5)
    trainer_kwargs.setdefault("enable_progress_bar", False)
    trainer_kwargs.setdefault("enable_model_summary", False)
    trainer_kwargs.setdefault("logger", False)
    trainer_kwargs.setdefault("enable_checkpointing", False)
    trainer_kwargs["accelerator"] = "cpu"
    trainer_kwargs["devices"] = world_size
    trainer_kwargs["num_nodes"] = 1
    # gloo on CPU; the env vars from torchrun pre-init the group.
    trainer_kwargs["strategy"] = "ddp"

    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=dm)

    return {
        "rank": rank,
        "world_size": world_size,
        "ok": True,
        "error": None,
        "trainer_global_rank": int(trainer.global_rank),
        "trainer_world_size": int(trainer.world_size),
        "is_ddp_strategy": type(trainer.strategy).__name__,
        "ids": list(collected_ids),
    }


def _write_result(result_dir: Path, rank: int, payload: Dict[str, Any]) -> None:
    path = result_dir / f"rank_{rank}.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f)
    tmp.replace(path)


def main() -> int:
    args = _parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        _write_result(
            args.result_dir,
            rank,
            {
                "rank": rank,
                "world_size": world_size,
                "ok": False,
                "error": f"Invalid --config-json: {exc}",
                "ids": [],
            },
        )
        return 2

    try:
        payload = _run_one_rank(args.result_dir, config)
    except Exception:
        payload = {
            "rank": rank,
            "world_size": world_size,
            "ok": False,
            "error": traceback.format_exc(),
            "ids": [],
        }

    _write_result(args.result_dir, rank, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
