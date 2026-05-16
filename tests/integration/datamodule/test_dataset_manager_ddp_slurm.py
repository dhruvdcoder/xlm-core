"""SLURM-marked integration tests for ``DatasetManager``.

These tests submit a real multi-GPU job via ``sbatch --wait`` and
collect per-rank result files written by the SLURM script.  They are
gated by both the ``integration`` and ``slurm`` markers and are *also*
auto-skipped unless the user explicitly opts in via the environment
variable ``XLM_INTEGRATION_SLURM_ENABLE=1``.  This double gate means
the suite stays inert on developer laptops and on cluster login nodes
that happen to have ``sbatch`` on PATH but no usable partition.

How to run
==========

Minimal::

    XLM_INTEGRATION_SLURM_ENABLE=1 \\
    pytest -m "integration and slurm" tests/integration/datamodule/

If the default partition / QoS in ``script.sh`` does not work on your
cluster, override them via :envvar:`XLM_INTEGRATION_SBATCH_ARGS`
(comma-separated)::

    XLM_INTEGRATION_SLURM_ENABLE=1 \\
    XLM_INTEGRATION_SBATCH_ARGS="--partition=gpu,--qos=debug" \\
    pytest -m "integration and slurm" tests/integration/datamodule/

The values land verbatim in the ``sbatch`` argv list (split on
``,``), so any flag understood by ``sbatch`` is fair game.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.integration._slurm import have_sbatch, submit_sbatch_and_wait


pytestmark = [pytest.mark.integration, pytest.mark.slurm]


SCENARIO_DIR = (
    Path(__file__).resolve().parent / "slurm" / "ddp_iterable_shards"
)
SCRIPT_SH = SCENARIO_DIR / "script.sh"


def _extra_sbatch_args() -> list:
    raw = os.environ.get("XLM_INTEGRATION_SBATCH_ARGS", "").strip()
    if not raw:
        return []
    return [a.strip() for a in raw.split(",") if a.strip()]


@pytest.fixture(autouse=True)
def _skip_unless_enabled():
    if not have_sbatch():
        pytest.skip("sbatch not on PATH -- skipping SLURM integration tests")
    if os.environ.get("XLM_INTEGRATION_SLURM_ENABLE") != "1":
        pytest.skip(
            "Set XLM_INTEGRATION_SLURM_ENABLE=1 to run SLURM-submitting tests"
        )


def test_ddp_iterable_shards_full_coverage(result_dir: Path):
    """Two-rank SLURM job covers the dataset without overlap."""
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
        "run": {"max_batches_per_epoch": 1000, "seed": 0},
    }
    results = submit_sbatch_and_wait(
        script_sh=SCRIPT_SH,
        result_dir=result_dir,
        config=config,
        expected_world_size=2,
        timeout=900.0,
        extra_sbatch_args=_extra_sbatch_args(),
    )
    assert len(results) == 2
    rank0_ids = list(results[0]["epochs"][0]["ids"])
    rank1_ids = list(results[1]["epochs"][0]["ids"])
    assert set(rank0_ids).isdisjoint(set(rank1_ids))
    assert sorted(rank0_ids + rank1_ids) == list(range(1000, 1060))
    for r in results:
        assert r["is_iterable_dataset"] is True
