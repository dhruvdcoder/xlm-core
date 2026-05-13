"""SLURM job submission helpers for ``slurm``-marked integration tests.

The CPU multi-process runner (``_runner.py``) is sufficient for most
DDP correctness checks.  This module is for the small number of tests
that need a *real* multi-GPU job submitted via ``sbatch`` -- typically
to catch NCCL- or fabric-specific bugs that gloo on CPU cannot
reproduce.

Public surface
==============

* :func:`have_sbatch` -- True iff ``sbatch`` is on PATH.  Tests should
  ``pytest.skip`` when this is False so the suite stays runnable on
  developer laptops.
* :func:`submit_sbatch_and_wait` -- wraps ``sbatch --wait`` with a
  result-directory convention: per-rank JSON files written by the job
  are loaded and returned exactly like
  :func:`tests.integration._runner.run_cpu_distributed`.

Job-script convention
=====================

A SLURM script for this suite must:

1. Take the result directory as its *first* positional argument.
2. (Optional) Take a JSON-encoded config blob as its *second*
   positional argument and forward it to the python entrypoint via
   ``--config-json``.
3. Invoke the python entrypoint under ``srun``, passing
   ``--result-dir`` and ``--config-json`` so the script writes
   ``rank_<RANK>.json`` files there.
4. Exit with the entrypoint's exit code.

See ``tests/integration/datamodule/slurm/ddp_iterable_shards/script.sh``
for a concrete example.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def have_sbatch() -> bool:
    """Return True iff a SLURM ``sbatch`` binary is on PATH."""
    return shutil.which("sbatch") is not None


def submit_sbatch_and_wait(
    script_sh: Path,
    result_dir: Path,
    *,
    config: Optional[Mapping[str, Any]] = None,
    expected_world_size: int,
    timeout: float = 600.0,
    extra_sbatch_args: Optional[List[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Submit ``script_sh`` via ``sbatch --wait`` and collect rank result files.

    Args:
        script_sh: Path to the ``.sh`` SLURM job script.  Must accept
            ``<result_dir> [config_json]`` as positional arguments.
        result_dir: Directory the job will write ``rank_<RANK>.json``
            into.  Must already exist; ``--chdir`` is *not* set so the
            script sees this as an absolute path.
        config: Optional JSON-serialisable dict; serialised to a single
            JSON string and passed as the second positional arg.
        expected_world_size: Number of ranks the job will spawn.  The
            helper raises if any ``rank_<r>.json`` file is missing.
        timeout: Wall-clock timeout in seconds for the entire ``sbatch
            --wait`` call.  Make this generous: it includes queue time.
        extra_sbatch_args: Extra arguments inserted *before* the script
            path (e.g. ``["--qos=debug"]``).
        env: Extra environment variables for the ``sbatch`` call (does
            *not* propagate into the launched job unless the script
            forwards them).

    Returns:
        List of per-rank result dicts, sorted by ``rank``.

    Raises:
        AssertionError: If sbatch failed, or any expected
            ``rank_<r>.json`` file is missing, or any rank reported
            ``ok=False``.
        TimeoutExpired: If sbatch did not return within ``timeout``.
    """
    script_sh = Path(script_sh).resolve()
    result_dir = Path(result_dir).resolve()
    if not script_sh.exists():
        raise FileNotFoundError(f"SLURM script not found: {script_sh}")
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir does not exist: {result_dir}")

    cmd: List[str] = ["sbatch", "--wait", "--parsable"]
    if extra_sbatch_args:
        cmd.extend(extra_sbatch_args)
    cmd.append(str(script_sh))
    cmd.append(str(result_dir))
    if config is not None:
        cmd.append(json.dumps(config))

    sbatch_env = dict(os.environ)
    if env:
        sbatch_env.update(env)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=sbatch_env,
    )

    if proc.returncode != 0:
        raise AssertionError(
            "sbatch --wait failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"returncode: {proc.returncode}\n"
            f"--- STDOUT ---\n{proc.stdout}\n"
            f"--- STDERR ---\n{proc.stderr}\n"
            f"Inspect the SLURM logs in {result_dir} (slurm-*.out)."
        )

    results: List[Dict[str, Any]] = []
    missing: List[int] = []
    for rank in range(expected_world_size):
        path = result_dir / f"rank_{rank}.json"
        if not path.exists():
            missing.append(rank)
            continue
        with path.open() as f:
            results.append(json.load(f))
    if missing:
        raise AssertionError(
            f"SLURM job left rank result files missing: {missing}.\n"
            f"sbatch stdout: {proc.stdout!r}\n"
            f"Inspect the SLURM logs in {result_dir}."
        )

    results.sort(key=lambda r: r["rank"])

    failed = [r for r in results if not r.get("ok", False)]
    if failed:
        details = "\n".join(
            f"  rank {r['rank']}: {r.get('error', '<no error message>')}"
            for r in failed
        )
        raise AssertionError(
            "One or more SLURM ranks reported failure:\n" f"{details}"
        )

    return results
