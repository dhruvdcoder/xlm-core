"""End-to-end CPU Lightning Trainer DDP test for ``DatasetManager``.

This is the *tiered* Lightning case: every other DDP correctness
assertion lives in
``test_dataset_manager_ddp_cpu.py`` and runs against
:class:`xlm.datamodule.DatasetManager` directly.  The single test in
this file confirms the integration with the Lightning Trainer:

* ``Trainer(strategy="ddp", accelerator="cpu", devices=N)`` correctly
  threads ``rank`` / ``world_size`` through
  :class:`xlm.datamodule.TextDataModule` into
  :meth:`DatasetManager.setup` and :meth:`DatasetManager.get_dataloader`;
* the iterable + DDP path produces non-overlapping per-rank coverage
  inside an actual ``training_step`` loop -- the same path that breaks
  in production when ``split_dataset_by_node`` is mis-wired.

Subprocess entrypoint:
:mod:`tests.integration._scripts.ddp_lightning_entrypoint`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration._runner import run_cpu_distributed


pytestmark = [pytest.mark.integration, pytest.mark.ddp]


ENTRYPOINT = "tests.integration._scripts.ddp_lightning_entrypoint"


def test_lightning_ddp_iterable_threads_rank_world_size(result_dir: Path):
    """Trainer-driven iterable DDP: per-rank coverage is non-overlapping."""
    config = {
        "dsm_kwargs": {
            "full_name": "mem/raw_large/train",
            "iterable_dataset_shards": 4,
            "dataloader_kwargs": {
                "batch_size": 4,
                "num_workers": 1,
                "pin_memory": False,
            },
        },
        "trainer_kwargs": {
            # 4 batches/rank * 4 batch_size = 16 ids/rank; with world=2
            # that's 32 of the 60 examples -- large enough to detect any
            # split-by-node misrouting without overflowing one epoch.
            "max_epochs": 1,
            "limit_train_batches": 4,
            "seed": 0,
        },
    }
    results = run_cpu_distributed(
        script_module=ENTRYPOINT,
        world_size=2,
        result_dir=result_dir,
        config=config,
        timeout=240.0,
    )
    assert len(results) == 2

    for r in results:
        # Lightning must report consistent rank metadata.
        assert r["trainer_global_rank"] == r["rank"]
        assert r["trainer_world_size"] == r["world_size"] == 2
        # We requested DDP; sanity-check Lightning honoured it.
        assert "DDP" in r["is_ddp_strategy"], r["is_ddp_strategy"]
        # Each rank consumed batch_size * limit_train_batches examples.
        assert len(r["ids"]) == 4 * 4

    # Per-rank id sets must not overlap (split_dataset_by_node).
    rank0_ids = set(results[0]["ids"])
    rank1_ids = set(results[1]["ids"])
    assert rank0_ids.isdisjoint(rank1_ids), (
        f"Lightning DDP coverage overlap: rank0={rank0_ids & rank1_ids}"
    )
    # All ids fall inside the in-memory dataset's id range.
    for ids in (rank0_ids, rank1_ids):
        assert all(1000 <= i < 1000 + 60 for i in ids)
