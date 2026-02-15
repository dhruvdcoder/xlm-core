"""End-to-end smoke tests for model-dataset combinations.

Each entry in :data:`SMOKE_RUNS` is executed as a full ``xlm`` CLI
invocation with the ``debug=smoke`` config (defined in
``tests/configs/debug/smoke.yaml``).  This runs 5 training steps with
validation at step 3 on CPU -- just enough to exercise the full
train + val pipeline without requiring a GPU or significant time.

**To add a new smoke test**, append a ``(experiment, job_type)`` tuple
to :data:`SMOKE_RUNS` below.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Absolute path to ``tests/configs/`` -- passed to ``--config-dir``
#: so that Hydra can discover ``debug/smoke.yaml``.
TESTS_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

#: Each tuple is ``(experiment_config, job_type)``.
#: Add new model-dataset combos here -- one line each.
SMOKE_RUNS: list[tuple[str, str]] = [
    ("star_easy_mlm", "train"),
    ("star_easy_arlm", "train"),
    ("star_easy_ilm", "train"),
    # ("star_easy_mdlm", "train"),  # uncomment when mdlm noise schedule is CI-ready
]

#: CLI overrides applied to every smoke run (on top of debug=smoke).
_COMMON_OVERRIDES: list[str] = [
    "debug=smoke",
    "trainer_strategy=cpu",
]


def _unique_experiments() -> list[str]:
    """Deduplicated experiment names from :data:`SMOKE_RUNS`."""
    seen: set[str] = set()
    result: list[str] = []
    for exp, _ in SMOKE_RUNS:
        if exp not in seen:
            seen.add(exp)
            result.append(exp)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _xlm_bin():
    """Session-scoped path to the ``xlm`` console-script."""
    path = shutil.which("xlm")
    if path is None:
        pytest.skip("xlm console script not found on PATH")
    return path


@pytest.fixture(scope="session")
def prepare_smoke_data(_xlm_bin, tmp_path_factory):
    """Run ``prepare_data`` once per unique experiment in :data:`SMOKE_RUNS`.

    Respects ``DATA_DIR``, ``HF_HOME``, and ``HF_DATASETS_CACHE`` env vars
    so that pre-downloaded data is reused automatically.
    """
    output_dir = tmp_path_factory.mktemp("prepare_data")
    for experiment in _unique_experiments():
        cmd = [
            _xlm_bin,
            "--config-dir",
            TESTS_CONFIG_DIR,
            "job_type=prepare_data",
            f"job_name=prepare_{experiment}",
            f"experiment={experiment}",
            "debug=smoke",
            "trainer_strategy=cpu",
            f"paths.log_dir={output_dir}",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            pytest.fail(
                f"prepare_data failed for {experiment}:\n"
                f"STDOUT (last 2000 chars):\n{result.stdout[-2000:]}\n\n"
                f"STDERR (last 2000 chars):\n{result.stderr[-2000:]}"
            )


# ---------------------------------------------------------------------------
# Parametrized smoke test
# ---------------------------------------------------------------------------


@pytest.mark.cli
@pytest.mark.slow
@pytest.mark.parametrize(
    "experiment,job_type",
    SMOKE_RUNS,
    ids=[f"{exp}-{jt}" for exp, jt in SMOKE_RUNS],
)
def test_smoke_run(
    experiment: str,
    job_type: str,
    _xlm_bin: str,
    tmp_path: Path,
    prepare_smoke_data,  # noqa: ARG001 -- ensures data is ready
):
    """Run a short end-to-end smoke test for *experiment* / *job_type*.

    Success criteria: the ``xlm`` process exits with return code 0.
    """
    cmd = [
        _xlm_bin,
        "--config-dir",
        TESTS_CONFIG_DIR,
        f"job_type={job_type}",
        f"job_name=test_{experiment}",
        f"experiment={experiment}",
        f"paths.log_dir={tmp_path}",
        *_COMMON_OVERRIDES,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Smoke run failed for {experiment} ({job_type}):\n"
        f"Command: {' '.join(cmd)}\n\n"
        f"STDOUT (last 2000 chars):\n{result.stdout[-2000:]}\n\n"
        f"STDERR (last 2000 chars):\n{result.stderr[-2000:]}"
    )
