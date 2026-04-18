"""CPU multi-process distributed runner for integration tests.

This module hides every detail of launching a multi-process PyTorch
distributed job from the test files.  The same launcher is used by all
``ddp``-marked CPU integration tests; SLURM-marked tests use a separate
helper (see :mod:`tests.integration._slurm`).

Usage from a test::

    results = run_cpu_distributed(
        script_module="tests.integration._scripts.ddp_dsm_entrypoint",
        world_size=2,
        result_dir=result_dir,
        config={"dsm_kwargs": {...}, "setup_kwargs": {...}, "run": {...}},
    )

Each rank in the spawned job writes a single ``rank_<r>.json`` file into
``result_dir`` describing what it observed.  The runner then loads the
files, sorts them by rank and returns the parsed list of dicts.

Design notes
============

* We use ``python -m torch.distributed.run`` (a.k.a. ``torchrun``) rather
  than spawning processes by hand so the entrypoint receives the standard
  ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` / ``MASTER_ADDR`` /
  ``MASTER_PORT`` environment variables.  The entrypoint then calls
  ``torch.distributed.init_process_group(backend="gloo")`` which is the
  CPU-friendly backend supported on every CI runner.
* ``MASTER_PORT`` is selected at launch time by binding a socket to
  port 0 and reading back the kernel-assigned port.  This keeps the
  runner safe to run in parallel (xdist) on the same host.
* The launched python interpreter is the *current* one (``sys.executable``)
  so the subprocess inherits the active virtualenv / conda env without
  any extra plumbing.  ``PYTHONPATH`` is augmented with the workspace
  ``src`` and ``tests`` roots so the entrypoint can import both
  ``xlm.*`` and ``tests.integration.*``.
* The runner enforces a wall-clock timeout (default 120 s) and, on
  failure, surfaces both the subprocess exit code and the captured
  stderr to make pytest tracebacks self-contained.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
TESTS_ROOT = REPO_ROOT / "tests"


def _pick_free_port() -> int:
    """Return a TCP port currently free on the loopback interface.

    There is an unavoidable TOCTOU race (the kernel may hand the same
    port to a different process between the close() and torchrun's bind)
    but in practice this is enough for serial CI usage and matches what
    HuggingFace's ``test_distributed`` does.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _build_env(extra_env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Build the environment for the spawned distributed processes."""
    env = dict(os.environ)
    # Make sure the entrypoint can import xlm.* and tests.integration.*.
    pythonpath_parts: List[str] = []
    for p in (str(SRC_ROOT), str(REPO_ROOT)):
        if p not in pythonpath_parts:
            pythonpath_parts.append(p)
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    # Avoid runaway BLAS thread fan-out when many ranks share one CPU.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    # Suppress HF datasets progress bars in subprocess output.
    env.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    if extra_env:
        env.update(extra_env)
    return env


def run_cpu_distributed(
    script_module: str,
    world_size: int,
    result_dir: Path,
    *,
    config: Optional[Mapping[str, Any]] = None,
    extra_args: Optional[List[str]] = None,
    timeout: float = 120.0,
    env: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Launch ``script_module`` under ``torch.distributed.run`` and collect results.

    Args:
        script_module: Dotted module path of an entrypoint script that
            (a) initialises a distributed process group with the gloo
            backend, (b) writes ``rank_<RANK>.json`` into ``result_dir``,
            and (c) calls ``dist.destroy_process_group()`` on exit.
        world_size: Number of ranks to spawn.  Each rank is a separate
            python process on this host.
        result_dir: Directory the entrypoint writes per-rank result files
            to.  Must already exist (the ``result_dir`` pytest fixture
            handles this).
        config: Optional JSON-serialisable dict forwarded to the
            entrypoint via ``--config-json``.  The entrypoint is free to
            interpret the schema however it likes.
        extra_args: Optional extra CLI arguments appended after the
            ``--config-json`` block.
        timeout: Wall-clock timeout in seconds for the whole job.
        env: Optional extra environment variables for the subprocess.

    Returns:
        List of per-rank result dicts, sorted by ``rank``.

    Raises:
        AssertionError: If the subprocess exited with a non-zero status,
            timed out, or did not write the expected
            ``rank_<r>.json`` files.
    """
    result_dir = Path(result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir does not exist: {result_dir}")

    master_port = _pick_free_port()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=127.0.0.1",
        f"--master_port={master_port}",
        "-m",
        script_module,
        "--result-dir",
        str(result_dir),
    ]
    if config is not None:
        cmd.extend(["--config-json", json.dumps(config)])
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(
        cmd,
        env=_build_env(env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=str(REPO_ROOT),
    )

    if proc.returncode != 0:
        raise AssertionError(
            "Distributed runner failed with exit code "
            f"{proc.returncode}.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"--- STDOUT ---\n{proc.stdout}\n"
            f"--- STDERR ---\n{proc.stderr}\n"
        )

    results: List[Dict[str, Any]] = []
    missing: List[int] = []
    for rank in range(world_size):
        path = result_dir / f"rank_{rank}.json"
        if not path.exists():
            missing.append(rank)
            continue
        with path.open() as f:
            results.append(json.load(f))
    if missing:
        raise AssertionError(
            f"Missing rank result files for ranks {missing} in {result_dir}.\n"
            f"--- STDOUT ---\n{proc.stdout}\n"
            f"--- STDERR ---\n{proc.stderr}\n"
        )

    results.sort(key=lambda r: r["rank"])

    failed = [r for r in results if not r.get("ok", False)]
    if failed:
        details = "\n".join(
            f"  rank {r['rank']}: {r.get('error', '<no error message>')}"
            for r in failed
        )
        raise AssertionError(
            "One or more ranks reported failure:\n"
            f"{details}\n"
            f"--- STDOUT ---\n{proc.stdout}\n"
            f"--- STDERR ---\n{proc.stderr}\n"
        )

    return results
