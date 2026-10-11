"""CPU (gloo) FSDP regression test for optimizer coverage of the root unit.

Background (issue #59): Lightning's FSDPStrategy wraps the whole Harness.
With a per-block auto_wrap_policy every parameter outside a block
(embeddings, final norm, lm_head, fresh heads) lands in the *root* FSDP unit,
whose FlatParameter is registered on the wrapper above harness.model. With
use_orig_params=False, harness.model.parameters() therefore misses it and the
optimizer silently never updates those weights.

These tests wrap a tiny model with raw FullyShardedDataParallel on CPU (the
Lightning strategy itself needs CUDA; see the SLURM test for the end-to-end
path) and check that

* the legacy harness.model.parameters() source is caught by
  Harness.check_optimizer_covers_trainable_params when use_orig_params=False
  (and that the root weights really do not move in that configuration);
* Harness.optimizer_named_parameters() covers every trainable parameter and
  the root weights move after one step, for both use_orig_params settings;
* FSDPHarness.configure_gradient_clipping clips the *global* norm.

Run with::

    pytest -m "integration and ddp" tests/integration/test_fsdp_optimizer_coverage.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration._runner import run_cpu_distributed

pytestmark = [pytest.mark.integration, pytest.mark.ddp]

ENTRYPOINT = "tests.integration._scripts.fsdp_optimizer_coverage_entrypoint"
WORLD_SIZE = 2


def _run(result_dir: Path, use_orig_params: bool):
    return run_cpu_distributed(
        script_module=ENTRYPOINT,
        world_size=WORLD_SIZE,
        result_dir=result_dir,
        config={"use_orig_params": use_orig_params},
        timeout=600.0,
    )


def test_legacy_model_parameters_misses_root_unit_without_orig_params(
    result_dir: Path,
):
    """use_orig_params=False + harness.model.parameters() -> check raises, root frozen."""
    results = _run(result_dir, use_orig_params=False)
    for r in results:
        legacy = r["legacy"]
        assert legacy["optimizer_numel"] < r["trainable_numel"], r
        assert legacy["raised"], legacy
        assert "_flat_param" in legacy["error_snippet"], legacy["error_snippet"]
        assert "use_orig_params" in legacy["error_snippet"]
        # The silent failure mode the check exists to prevent.
        assert legacy["root_moved"] is False, legacy


@pytest.mark.parametrize("use_orig_params", [False, True])
def test_optimizer_named_parameters_covers_root_unit(
    result_dir: Path, use_orig_params: bool
):
    """Harness.optimizer_named_parameters() covers all trainable params."""
    results = _run(result_dir, use_orig_params=use_orig_params)
    for r in results:
        fixed = r["fixed"]
        assert fixed["optimizer_numel"] == r["trainable_numel"], r
        assert not fixed["raised"], fixed["error_snippet"]
        assert fixed["root_moved"] is True, fixed
        # Wrapper prefixes are stripped so names look like plain model names.
        for name in fixed["names"]:
            assert "_fsdp_wrapped_module" not in name, name
            assert not name.startswith("model."), name
        if use_orig_params:
            assert "embed.weight" in fixed["names"], fixed["names"]
            assert "head.weight" in fixed["names"], fixed["names"]
            # With original params the legacy path is complete as well.
            assert r["legacy"]["optimizer_numel"] == r["trainable_numel"]
            assert not r["legacy"]["raised"]
            assert r["legacy"]["root_moved"] is True
        else:
            assert "_flat_param" in fixed["names"], fixed["names"]


@pytest.mark.parametrize("use_orig_params", [False, True])
def test_fsdp_harness_clips_global_grad_norm(
    result_dir: Path, use_orig_params: bool
):
    results = _run(result_dir, use_orig_params=use_orig_params)
    for r in results:
        clip = r["clip"]
        assert clip["norm_before"] > 0.5, clip
        assert clip["norm_after"] == pytest.approx(0.5, rel=1e-3), clip
    # Every rank computes the same all-reduced global norm.
    norms = {round(r["clip"]["norm_after"], 6) for r in results}
    assert len(norms) == 1, norms
