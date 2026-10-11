"""SLURM-marked 2-GPU end-to-end test for FSDP optimizer coverage (issue #59).

Each test submits one real 2-GPU job via sbatch --wait that drives the full
xlm path: Hydra composition (experiment=star_easy_mlm debug=smoke
trainer_strategy=fsdp + a per-block auto_wrap_policy), Lightning's
FSDPStrategy wrapping the Harness, Harness.configure_optimizers running
after the wrap, and five smoke training steps. A probe callback records
per-rank evidence (see slurm/fsdp_optimizer_coverage/probe.py).

The CPU gloo test in tests/integration/test_fsdp_optimizer_coverage.py
covers the helper semantics; this file covers the Lightning ordering and the
inherited config that the bug actually depends on.

How to run (from the repo root, inside the project env, on a login node)::

    XLM_INTEGRATION_SLURM_ENABLE=1 \\
    XLM_INTEGRATION_SBATCH_ARGS="--partition=gpu,--constraint=a100|l40s" \\
    pytest -m "integration and slurm" tests/integration/fsdp/ -s

Jobs take a few minutes each plus queue time; ``timeout`` below is generous.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.integration._slurm import have_sbatch, submit_sbatch_and_wait

pytestmark = [pytest.mark.integration, pytest.mark.slurm]

SCENARIO_DIR = Path(__file__).resolve().parent / "slurm" / "fsdp_optimizer_coverage"
SCRIPT_SH = SCENARIO_DIR / "script.sh"
REPO_ROOT = Path(__file__).resolve().parents[3]
WORLD_SIZE = 2
SMOKE_STEPS = 5  # tests/configs/debug/smoke.yaml: trainer.max_steps


def _extra_sbatch_args() -> List[str]:
    raw = os.environ.get("XLM_INTEGRATION_SBATCH_ARGS", "").strip()
    if not raw:
        return []
    return [a.strip() for a in raw.split(",") if a.strip()]


@pytest.fixture(autouse=True)
def _skip_unless_enabled():
    if not have_sbatch():
        pytest.skip("sbatch not on PATH -- skipping SLURM integration tests")
    if os.environ.get("XLM_INTEGRATION_SLURM_ENABLE") != "1":
        pytest.skip("Set XLM_INTEGRATION_SLURM_ENABLE=1 to run SLURM-submitting tests")


def _submit(result_dir: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = submit_sbatch_and_wait(
        script_sh=SCRIPT_SH,
        result_dir=result_dir,
        config=config,
        expected_world_size=WORLD_SIZE,
        timeout=1800.0,
        extra_sbatch_args=_extra_sbatch_args(),
        # sbatch spools the script, so it cannot locate the repo from
        # BASH_SOURCE; forward the root explicitly (sbatch exports env).
        env={"XLM_REPO_ROOT": str(REPO_ROOT)},
    )
    assert len(results) == WORLD_SIZE
    print(f"\n[fsdp slurm] results in {result_dir}")
    return results


def _assert_trained_and_covered(results: List[Dict[str, Any]]) -> None:
    for r in results:
        assert r["train_error"] is None, (
            f"rank {r['rank']} raised {r['train_error_type']}:\n{r['train_error']}"
        )
        probe = r["probe"]
        assert probe["is_fsdp"] is True, probe
        assert probe["num_root_unit_params"] >= 1, probe
        assert probe["coverage_ok"] is True, probe["coverage_error"]
        assert probe["optimizer_numel"] == probe["trainable_numel"], probe
        assert probe["global_step"] == SMOKE_STEPS, probe["global_step"]
        # Embedding / output layer live in the root unit and must move.
        assert probe["root_moved"] is True, probe


def test_fsdp_harness_default_config_trains_all_parameters(result_dir: Path):
    """FSDPHarness + inherited trainer_strategy/fsdp.yaml (use_orig_params=true)."""
    results = _submit(
        result_dir,
        {
            "harness": "xlm.harness.FSDPHarness",
            "use_orig_params": True,
            "gradient_clip_val": 1.0,
        },
    )
    _assert_trained_and_covered(results)
    for r in results:
        probe = r["probe"]
        assert probe["harness_class"] == "FSDPHarness"
        assert probe["use_orig_params"] is True
        # FSDPHarness logs the all-reduced global norm, not the local shard.
        logged = probe["logged_total_grad_norms"]
        global_norms = probe["global_grad_norms"]
        local_norms = probe["local_grad_norms"]
        assert len(logged) >= 1 and len(global_norms) >= 1, probe
        for lg, gl in zip(logged, global_norms):
            assert lg == pytest.approx(gl, rel=1e-3), (lg, gl)
        assert any(
            abs(lo - gl) > 1e-6 * max(1.0, gl) for lo, gl in zip(local_norms, global_norms)
        ), "local shard norm equals global norm; sharding did not happen?"


def test_base_harness_with_orig_params_trains_all_parameters(result_dir: Path):
    """Plain Harness + use_orig_params=true (new default). Clipping disabled
    because Lightning's FSDPPrecision rejects norm clipping without FSDPHarness."""
    results = _submit(
        result_dir,
        {
            "harness": "xlm.harness.Harness",
            "use_orig_params": True,
            "gradient_clip_val": None,
        },
    )
    _assert_trained_and_covered(results)
    for r in results:
        assert r["probe"]["harness_class"] == "Harness"


def test_base_harness_without_orig_params_fails_fast(result_dir: Path):
    """use_orig_params=false + MLM's model-level param groups: the root unit is
    missed and the coverage check must raise in configure_optimizers instead
    of training a subset silently (the original bug)."""
    results = _submit(
        result_dir,
        {
            "harness": "xlm.harness.Harness",
            "use_orig_params": False,
            "gradient_clip_val": None,
        },
    )
    for r in results:
        assert r["train_error_type"] == "RuntimeError", (
            f"rank {r['rank']}: expected coverage RuntimeError, got "
            f"{r['train_error_type']}:\n{r['train_error']}"
        )
        message = r.get("train_error_message", "") or r["train_error"]
        assert "not in any optimizer param group" in message, message
        assert "_flat_param" in message, message
        # The run never reached on_fit_start.
        assert "global_step" not in r["probe"], r["probe"]
