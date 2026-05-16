# Integration tests

Integration tests for `xlm` exercise multi-component flows that unit tests cannot reach: real DDP multi-process behaviour, end-to-end `DatasetManager` lifecycles, and `lightning.Trainer` runs. They live under `tests/integration/` and are driven by their own set of pytest markers (`integration`, `ddp`, `slurm`).

The exhaustive single-method matrix for `DatasetManager` (every branch of `__init__` / `prepare_data` / `setup` / `get_dataloader`) lives in `tests/core/test_datamodule.py` — it shares the same in-memory registry, monkey-patch and fixtures, but each test stays in a single process so it runs as part of the fast unit suite.

This page documents the **architecture**: what each piece does, how they fit together, and how to add new scenarios. Command-line cheatsheets also appear under **[Running tests → Integration tests](running-tests.md#integration-tests)**; the detailed tiers are below under [How to run integration tests](#how-to-run-integration-tests).

## How to run integration tests

> **Looking for the per-method `DatasetManager` tests?** Every branch of
> `__init__` / `prepare_data` / `setup` / `get_dataloader` is covered by
> `tests/core/test_datamodule.py` (single-process, ~1 s, runs as part of the
> fast unit suite). The integration suite covers what those tests cannot
> exercise in a single process.

The integration suite is split into three tiers controlled by markers:

| Tier | Marker selector | What it does | Typical runtime |
| --- | --- | --- | --- |
| Single-process lifecycle | `integration and not ddp` | End-to-end `prepare_data -> setup -> get_dataloader -> iterate` for both map-style and iterable backends in one Python process. | ~1 s |
| CPU multi-process DDP | `integration and ddp and not slurm` | Spawns `world_size` processes via `torch.distributed.run --backend=gloo`; covers `split_dataset_by_node`, `StatefulDistributedSampler`, `set_epoch`, `make_infinite`, plus one CPU Lightning Trainer DDP run. | ~2 min |
| SLURM multi-GPU DDP | `integration and slurm` | Submits a real GPU `nccl` job via `sbatch --wait`. Opt-in: see below. | depends on queue |

### Run the single-process lifecycle tier

```bash
pytest -m "integration and not ddp" tests/integration/
```

Fast (~1 s), no subprocesses; safe to run on every push.

### Run the CPU multi-process DDP tier

```bash
pytest -m "integration and ddp and not slurm" tests/integration/
```

Spawns Python subprocesses with `gloo`. Expect ~10-15 s per test.
Requires nothing beyond the standard test environment.

### Run the SLURM tier (multi-GPU)

The SLURM tests are double-gated: they auto-skip without `sbatch` on
PATH and *also* without `XLM_INTEGRATION_SLURM_ENABLE=1`. This
prevents accidental job submissions on cluster login nodes.

```bash
XLM_INTEGRATION_SLURM_ENABLE=1 \
pytest -m "integration and slurm" tests/integration/
```

If the default partition / GRES in
`tests/integration/datamodule/slurm/ddp_iterable_shards/script.sh`
does not match your cluster, override via
`XLM_INTEGRATION_SBATCH_ARGS` (comma-separated, forwarded verbatim
to `sbatch`):

```bash
XLM_INTEGRATION_SLURM_ENABLE=1 \
XLM_INTEGRATION_SBATCH_ARGS="--partition=gpu,--qos=debug" \
pytest -m "integration and slurm" tests/integration/
```

Per-rank `rank_<RANK>.json` result files and SLURM `slurm-*.out` logs
land in pytest's `tmp_path`; on failure the assertion message points at the directory.

### Run the entire integration suite locally

```bash
pytest -m "integration and not slurm" tests/integration/
```

This runs everything that does not require SLURM (~2 min).

## Goals and design choices

1. **No network I/O.** Every test reads from a small in-memory dataset registry (`tests/datamodule_helpers.py::build_inmem_datasets`) and monkey-patches `DatasetManager._download` to return those datasets. CI can run the entire suite offline.
2. **Tiered scope.** Most assertions target `DatasetManager` directly (fast); a small number of end-to-end tests instantiate a `lightning.Trainer` to confirm `rank` / `world_size` propagate correctly.
3. **Two execution modes.** CPU multi-process via `torch.distributed.run` for fast / CI-friendly DDP coverage, plus an opt-in SLURM tier for real multi-GPU `nccl` jobs that catch fabric-specific bugs.
4. **Shared helpers and fixtures.** `tests/datamodule_helpers.py` (importable, fixture-free) and the `inmem_datasets` / `patched_download` / `dataset_manager_factory` fixtures in the root `tests/conftest.py` are used by both the fast unit suite *and* the integration suite. This avoids duplication and keeps the in-memory registry definitions in one place.
5. **Reusable infra, not one-off scripts.** All multi-process tests share one helper (`run_cpu_distributed`) and one job-submission helper (`submit_sbatch_and_wait`). Each subprocess writes a `rank_<RANK>.json` result file; the parent test parses these and makes per-rank assertions. Adding a new scenario means writing one new entrypoint script and one new test, not duplicating subprocess plumbing.

## Directory layout

```
tests/
├── conftest.py                 # root fixtures (tokenizer, model, plus
│                               # DatasetManager fixtures: inmem_datasets,
│                               # patched_download, dataset_manager_factory,
│                               # manual_cache_dir, result_dir, simple_collator)
├── datamodule_helpers.py       # importable, fixture-free helpers
│                               # (datasets registry, patch helpers,
│                               # IdTrackingCollator, processors)
├── core/
│   ├── ...
│   └── test_datamodule.py      # single-method matrix (fast, single-process)
└── integration/
    ├── __init__.py
    ├── _runner.py              # run_cpu_distributed: torch.distributed.run
    │                           # launcher + per-rank JSON collection
    ├── _slurm.py               # submit_sbatch_and_wait: sbatch --wait wrapper
    ├── _scripts/
    │   ├── __init__.py
    │   ├── ddp_dsm_entrypoint.py        # subprocess entrypoint for CPU
    │   │                                # multi-process DatasetManager tests
    │   └── ddp_lightning_entrypoint.py  # subprocess entrypoint for the
    │                                    # CPU Lightning Trainer DDP test
    └── datamodule/
        ├── __init__.py
        ├── test_dataset_manager_lifecycle.py    # end-to-end single-process
        ├── test_dataset_manager_ddp_cpu.py      # CPU multi-process DDP
        ├── test_dataset_manager_ddp_lightning_cpu.py
        │                                        # CPU Lightning Trainer DDP
        ├── test_dataset_manager_ddp_slurm.py    # SLURM-marked GPU DDP
        └── slurm/
            └── ddp_iterable_shards/
                ├── README.md
                ├── script.sh                    # sbatch --wait entrypoint
                └── script.py                    # per-rank GPU DDP body
```

## Architecture

```mermaid
flowchart LR
    subgraph Pytest["pytest process"]
        TestCore["tests/core/test_datamodule.py<br/>(single-method matrix)"]
        TestLife["test_dataset_manager_lifecycle.py<br/>(end-to-end single-process)"]
        TestCpu["test_dataset_manager_ddp_cpu.py<br/>(CPU multi-process)"]
        TestLight["test_dataset_manager_ddp_lightning_cpu.py<br/>(CPU Lightning DDP)"]
        TestSlurm["test_dataset_manager_ddp_slurm.py<br/>(SLURM, opt-in)"]

        Conftest["tests/conftest.py<br/>fixtures:<br/>inmem_datasets,<br/>patched_download,<br/>dataset_manager_factory,<br/>manual_cache_dir,<br/>result_dir"]
        Helpers["tests/datamodule_helpers.py<br/>build_inmem_datasets,<br/>patch_dataset_manager_download,<br/>example_to_input_ids,<br/>IdTrackingCollator,<br/>pack_with_id, ..."]
        Runner["_runner.py<br/>run_cpu_distributed"]
        Slurm["_slurm.py<br/>submit_sbatch_and_wait"]
    end

    subgraph CpuChildren["CPU subprocesses<br/>(torch.distributed.run --nproc_per_node=N, gloo)"]
        Entry1["ddp_dsm_entrypoint.py<br/>(rank 0..N-1)"]
        Entry2["ddp_lightning_entrypoint.py<br/>(rank 0..N-1)"]
    end

    subgraph SlurmJob["SLURM job<br/>(sbatch --wait, NCCL)"]
        Sh["script.sh<br/>sets MASTER_*<br/>+ srun"]
        Py["script.py<br/>(rank 0..N-1)"]
    end

    Result["result_dir/<br/>rank_0.json, rank_1.json, ..."]

    TestCore --> Conftest
    TestLife --> Conftest
    TestCpu --> Conftest
    TestLight --> Conftest
    TestSlurm --> Conftest
    Conftest --> Helpers

    TestCore -.->|in-process| Helpers
    TestLife -.->|in-process| Helpers

    TestCpu -->|launches| Runner
    Runner -->|spawns| CpuChildren
    Entry1 -->|writes| Result
    Entry2 -->|writes| Result

    TestLight -->|launches| Runner

    TestSlurm -->|launches| Slurm
    Slurm -->|sbatch| Sh
    Sh -->|srun python| Py
    Py -->|writes| Result

    Runner -->|reads| Result
    Slurm -->|reads| Result
    Result -->|parsed list[dict]<br/>per-rank| TestCpu
    Result -->|...| TestLight
    Result -->|...| TestSlurm

    CpuChildren -.->|monkey-patch _download| Helpers
    Py -.->|monkey-patch _download| Helpers
```

The single-process matrix in `tests/core/test_datamodule.py` and the end-to-end lifecycle test in `tests/integration/datamodule/` never spawn subprocesses; they run everything in the pytest process. Multi-process tests pay the subprocess startup cost once per test (`~5-10 s` for CPU, `queue_time + 30 s` for SLURM) in exchange for genuine DDP semantics.

## Subprocess result-file contract

Every subprocess entrypoint — CPU or SLURM, plain or Lightning —
writes a single `rank_<RANK>.json` file into the test's `result_dir`
with the same baseline schema:

```json
{
  "rank": 0,
  "world_size": 2,
  "ok": true,
  "error": null,
  "...": "scenario-specific fields (epochs, ids, batch_shapes, ...)"
}
```

If anything raises inside the entrypoint, the same file is still
written with `ok=false` and the textual traceback in `error`. Both
runners (`run_cpu_distributed`, `submit_sbatch_and_wait`) raise
`AssertionError` if any rank reports `ok=false`, so the parent test
sees a clean Python-level failure with the subprocess traceback
embedded in the message.

## Pytest markers

| Marker | Meaning | Defined in |
| --- | --- | --- |
| `integration` | Any test under `tests/integration/`. Implicitly slow (subprocess startup, dataset construction). | `pyproject.toml` |
| `ddp` | Test spawns multiple processes for distributed coverage (CPU `gloo` or GPU `nccl`). | `pyproject.toml` |
| `slurm` | Test submits a SLURM job via `sbatch`. Auto-skipped without `sbatch` *or* without `XLM_INTEGRATION_SLURM_ENABLE=1`. | `pyproject.toml` |

The markers compose:

```bash
pytest -m "integration and not ddp"     # single-process matrix only
pytest -m "integration and ddp"         # multi-process incl. Lightning
pytest -m "integration and slurm"       # SLURM only (needs opt-in env)
```

## Fixtures cheat-sheet

All fixtures live in the root `tests/conftest.py` so both the unit suite (`tests/core/`) and the integration suite (`tests/integration/`) can use them:

| Fixture | Scope | Purpose |
| --- | --- | --- |
| `inmem_datasets` | session | The canonical in-memory `{full_name: Dataset}` registry. |
| `patched_download` | function | Monkey-patches `DatasetManager._download` to read from the registry; returns a per-test mutable copy of it. |
| `manual_cache_dir` | function | Empty per-test directory passed as `manual_cache_dir` to `prepare_data` / `setup`. |
| `result_dir` | function | Empty per-test directory used by both runners as the per-rank JSON drop point. |
| `simple_collator` | function | `IdTrackingCollator` bound to `simple_tokenizer`. |
| `dataset_manager_factory` | function | Callable returning a fully-wired `DatasetManager` with sensible defaults; tests pass only the kwargs they want to override. |

`simple_tokenizer` is also defined in the root `tests/conftest.py`.

## In-memory dataset registry

`tests.datamodule_helpers.build_inmem_datasets()` returns:

| Key | Size | `id` range | Purpose |
| --- | --- | --- | --- |
| `mem/raw/train` | 17 | `0..16` | Default small dataset; matches HF's `test_distributed.py` sizing so split-by-node arithmetic is easy to reason about. |
| `mem/raw/val` | 7 | `100..106` | Eval dataloader tests. |
| `mem/raw/test` | 5 | `200..204` | Eval dataloader tests. |
| `mem/raw_large/train` | 60 | `1000..1059` | Iterable + DDP coverage tests; large enough for `num_shards=4 x num_workers=1 x world_size=2`. |

`id` values are globally unique across splits so DDP coverage / non-overlap can be asserted directly: every rank writes the `id` of every example it consumed, and the parent test simply checks that the union covers the expected range and the per-rank sets are disjoint.

## CPU multi-process runner

`tests.integration._runner.run_cpu_distributed`:

```python
results: list[dict] = run_cpu_distributed(
    script_module="tests.integration._scripts.ddp_dsm_entrypoint",
    world_size=2,
    result_dir=result_dir,           # the result_dir fixture
    config={"dsm_kwargs": {...}, "run": {...}},
    timeout=120.0,
)
```

What it does:

1. Picks a free TCP port on `127.0.0.1` for `MASTER_PORT`.
2. Builds a `python -m torch.distributed.run --nproc_per_node=N -m <script_module>`
   command using `sys.executable`, so the subprocess inherits the active conda / virtualenv automatically.
3. Augments `PYTHONPATH` with the workspace `src/` and root so `xlm.*` and `tests.*` are importable. Sets `OMP_NUM_THREADS=MKL_NUM_THREADS=1` to avoid BLAS thread storms.
4. Runs the subprocess, captures stdout / stderr, enforces a `timeout` wall-clock cap.
5. Parses every `rank_<r>.json` file from `result_dir` and returns the list sorted by rank. Raises a single `AssertionError` with both the subprocess output and the per-rank `error` strings on any failure.

## SLURM job runner

`tests.integration._slurm.submit_sbatch_and_wait`:

```python
results = submit_sbatch_and_wait(
    script_sh=Path(".../slurm/ddp_iterable_shards/script.sh"),
    result_dir=result_dir,
    config={"dsm_kwargs": {...}, "run": {...}},
    expected_world_size=2,
    timeout=900.0,
    extra_sbatch_args=["--partition=gpu"],
)
```

Same result-file contract as the CPU runner; the only differences are:

- `sbatch --wait --parsable` is used so the call blocks until the job hits a terminal state.
- `extra_sbatch_args` is forwarded verbatim to `sbatch`, so per-cluster knobs (partition, QoS, account, GRES overrides) live in the test invocation, not the script.
- The job script is responsible for setting `MASTER_ADDR` / `MASTER_PORT` / `RANK` / `WORLD_SIZE` / `LOCAL_RANK` from the SLURM allocation; see `tests/integration/datamodule/slurm/ddp_iterable_shards/script.sh` for the canonical pattern.

The SLURM tests are *double-gated*: even with `sbatch` on PATH, they auto-skip unless `XLM_INTEGRATION_SLURM_ENABLE=1` is set, so the suite stays inert on cluster login nodes.

## Adding a new test

### Single-method (fast, single-process)

Add a method to the existing classes in `tests/core/test_datamodule.py` (or a new class). Use `dataset_manager_factory(...)` from the root conftest, plus the `manual_cache_dir`, `simple_tokenizer`, `simple_collator`, and (if you need to mutate the registry) `patched_download` fixtures. The test runs as part of the regular unit suite — no `integration` marker needed.

### End-to-end single-process

For a full `prepare_data -> setup -> get_dataloader -> iterate`
sanity check, add to `tests/integration/datamodule/test_dataset_manager_lifecycle.py`.
This file *does* live under `tests/integration/` and is marked
`integration` because it touches every layer of `DatasetManager` end
to end (cache eviction, dataloader iteration, batch-shape contract).

### CPU multi-process DDP scenario

1. If the existing `ddp_dsm_entrypoint` config schema covers your scenario, just add a new test in `test_dataset_manager_ddp_cpu.py` that calls `run_cpu_distributed(...)` with the appropriate config.
2. If you need behaviour the entrypoint does not expose, extend `_scripts/ddp_dsm_entrypoint.py` with new config fields or write a new entrypoint script in `_scripts/`.
3. Assert on the parsed per-rank dicts — the subprocess does the work, the test does the math.

### SLURM scenario

1. Create a new directory under `tests/integration/datamodule/slurm/<scenario>/` containing `script.sh`, `script.py`, and a short `README.md`. Mirror the layout of `ddp_iterable_shards`.
2. Re-use `tests.datamodule_helpers.patch_dataset_manager_download` inside `script.py` so the SLURM job stays offline.
3. Add a `slurm`-marked pytest in `test_dataset_manager_ddp_slurm.py` that calls `submit_sbatch_and_wait(script_sh=<your script.sh>, ...)`.

### Lightning Trainer scenario

For end-to-end Trainer tests, prefer to extend `_scripts/ddp_lightning_entrypoint.py` rather than launching Lightning directly from the parent pytest process. The entrypoint already wires up `TextDataModule`, a no-op recorder `LightningModule`, and a CPU DDP Trainer; you typically only need to vary the `dsm_kwargs` / `trainer_kwargs` blocks.
