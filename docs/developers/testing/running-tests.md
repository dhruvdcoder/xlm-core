# Running tests

## Prerequisites

Install test tooling as described under [Test dependencies](../dependencies.md#test-dependencies).

## Test layout

```
tests/
├── configs/
│   └── debug/
│       └── smoke.yaml       # Hydra bundle for CLI smoke tests (use with --config-dir tests/configs)
├── conftest.py              # Shared fixtures (tokenizer, batches, tiny model kwargs,
│                            # plus DatasetManager fixtures: inmem_datasets,
│                            # patched_download, dataset_manager_factory, ...)
├── datamodule_helpers.py    # Importable helpers for DatasetManager tests
│                            # (in-memory dataset registry, patch helpers,
│                            # IdTrackingCollator, processors). Used by both
│                            # tests/core/ and tests/integration/.
├── core/                    # Unit tests for xlm core components
│   ├── test_tokenizers.py
│   ├── test_collators.py
│   ├── test_noise.py
│   ├── test_metrics.py
│   ├── test_datamodule.py   # Single-method DatasetManager matrix (fast)
│   └── test_harness.py
├── models/                  # Unit tests for each model family
│   ├── _base.py             # Base test mixin classes (BaseModelTests, BaseLossTests, …)
│   ├── mlm/                 # model, loss, predictor, collator
│   ├── mdlm/
│   ├── arlm/
│   └── ilm/
├── cli/                     # CLI / integration tests
│   ├── test_train.py
│   ├── test_eval.py
│   ├── test_generate.py
│   ├── test_scaffold.py
│   └── test_smoke.py        # End-to-end xlm subprocess smoke (slow)
└── integration/             # End-to-end and multi-process DatasetManager tests
    │                        # (lifecycle, CPU DDP, Lightning DDP, SLURM);
    │                        # see Integration tests docs for the architecture.
    ├── _runner.py           # run_cpu_distributed (torch.distributed.run launcher)
    ├── _slurm.py            # submit_sbatch_and_wait (sbatch --wait wrapper)
    ├── _scripts/            # Subprocess entrypoints for the CPU DDP runner
    └── datamodule/
        ├── test_dataset_manager_lifecycle.py        # Single-process end-to-end
        ├── test_dataset_manager_ddp_cpu.py          # CPU multi-process DDP
        ├── test_dataset_manager_ddp_lightning_cpu.py
        ├── test_dataset_manager_ddp_slurm.py        # GPU DDP via sbatch (opt-in)
        └── slurm/                                   # SLURM scenario sandboxes
```

Also see:

- **[Unit tests](unit-tests.md)** — writing tests, mixin pattern, adding a model family.

## End-to-end CLI smoke tests

These tests live in `tests/cli/test_smoke.py`. Each case runs the real `xlm` console script as a **subprocess** with:

- `--config-dir` pointing at `tests/configs`, so Hydra can resolve the `debug` group.
- `debug=smoke`, which loads `tests/configs/debug/smoke.yaml`: a minimal run (5 training steps, validation at step 3, batch size 1, CPU strategy, no checkpointing).
- `trainer_strategy=cpu` so runs do not require a GPU.

A session-scoped fixture first runs `job_type=prepare_data` for each unique experiment in `SMOKE_RUNS`, then the parametrized test runs `job_type=train` (or whatever you list). Success means the subprocess exits with code 0.

**Run smoke tests only** (requires `xlm` on your `PATH`, e.g. `pip install -e .`):

```bash
pytest tests/cli/test_smoke.py -m "cli and slow" -v
```

Markers `cli` and `slow` are defined in `pyproject.toml` under `[tool.pytest.ini_options]`.

**Caches:** point these at existing caches to avoid re-downloading datasets (the `prepare_data` step respects them):

```bash
export HF_HOME=/path/to/huggingface/cache
export HF_DATASETS_CACHE=/path/to/datasets/cache
export DATA_DIR=/path/to/local/data
```

**Add a new combo:** append a `(experiment_name, job_type)` tuple to `SMOKE_RUNS` in `tests/cli/test_smoke.py`. The parametrized test id becomes `experiment_name-job_type`.

## Running tests

### Run the fast unit tests (recommended during development)

```bash
pytest -m "not slow and not cli"
```

This runs only the fast tests (~76 tests) and skips anything that needs
large datasets, real noise schedules, Hydra configs, or subprocess calls.
It should complete in under 2 seconds.

### Run the full suite

```bash
pytest
```

This includes slow integration tests and CLI smoke tests.
Tests that require resources not yet available (tiny experiment configs,
real noise schedules, etc.) will be skipped automatically with a message
explaining what is needed.

### Run tests for a specific module

```bash
# All core tests
pytest tests/core/

# All MLM model tests
pytest tests/models/mlm/

# A single test file
pytest tests/models/arlm/test_loss_arlm.py

# A single test class or function
pytest tests/core/test_tokenizers.py::TestSimpleSpaceTokenizer::test_vocab_size
```

### Run only CLI tests

```bash
pytest -m cli
```

### Run only GPU tests

```bash
pytest -m gpu
```

## Markers

Tests use the following `pytest` markers (defined in `pyproject.toml`):

| Marker | Purpose                                                                                                     |
|--------|-------------------------------------------------------------------------------------------------------------|
| `slow` | Tests that are expensive (full training steps, large models, real datasets). Deselect with `-m "not slow"`. |
| `gpu`  | Tests that require a CUDA GPU.                                                                              |
| `cli`  | Tests that invoke the `xlm` CLI as a subprocess.                                                            |
| `integration` | Tests under `tests/integration/` (end-to-end lifecycle, CPU DDP, Lightning DDP, SLURM). Subprocess startup + dataset construction make them slower than unit tests. The fast single-method `DatasetManager` matrix lives in `tests/core/test_datamodule.py` and is **not** marked `integration`. |
| `ddp`  | Tests that spawn multiple processes for distributed coverage (CPU `gloo` or GPU `nccl`).                     |
| `slurm`| Tests that submit a real SLURM job. Auto-skipped unless `sbatch` is on PATH **and** `XLM_INTEGRATION_SLURM_ENABLE=1`. |

You can combine markers:

```bash
# Everything except slow and GPU tests
pytest -m "not slow and not gpu"
```

## Running with coverage

```bash
coverage run -m pytest -m "not slow and not cli"
coverage report
```

Coverage settings are in `pyproject.toml` under `[tool.coverage.*]`.
The current threshold is `fail_under = 90`.

## Integration tests

The `tests/integration/` suite exercises end-to-end `DatasetManager`
lifecycles, real DDP multi-process behaviour, and end-to-end
`lightning.Trainer` runs.

Full architecture, subprocess contract, tiered **`pytest`** invocation
cheatsheet, fixtures, runners, and how to add new scenarios are documented
under **[Integration tests](integration-tests.md)**.

> **Looking for the per-method `DatasetManager` tests?** Every branch of
> `__init__` / `prepare_data` / `setup` / `get_dataloader` is covered by
> `tests/core/test_datamodule.py` (single-process, ~1 s, runs as part of the
> fast unit suite). The integration suite covers what those tests cannot
> exercise in a single process.
