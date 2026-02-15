# Running Tests

## Prerequisites

1. **Install test dependencies** (if not already installed):

   ```bash
   pip install -r requirements/test_requirements.txt
   ```

   This installs `pytest`, `hypothesis`, and `coverage[toml]`.

2. **Install the package** in development mode so the `xlm` CLI is on
   your PATH:

   ```bash
   pip install -e .
   pip install -e ./xlm-models
   ```

## Test layout

```
tests/
├── conftest.py              # Shared fixtures
├── configs/                 # Hydra configs used only by tests
│   └── debug/
│       └── smoke.yaml       # Short-run debug config for smoke tests
└── cli/
    ├── test_smoke.py        # Parametrized end-to-end smoke tests
    └── test_scaffold.py     # xlm-scaffold CLI test
```

## Running tests

### Run all smoke tests

```bash
pytest tests/cli/test_smoke.py -v
```

This runs a short train + validation pass for every model-dataset
combination listed in `SMOKE_RUNS` (see below).  Each run executes
5 training steps with validation at step 3, on CPU, with batch size 1.

### Run a single combination

```bash
pytest "tests/cli/test_smoke.py::test_smoke_run[star_easy_mlm-train]" -v
```

### Run the scaffold test

```bash
pytest tests/cli/test_scaffold.py -v
```

## Data setup

Smoke tests download datasets automatically via a session-scoped
`prepare_data` fixture.  To speed this up or avoid re-downloading,
point the env vars to your existing cache before running pytest:

```bash
export HF_HOME=/path/to/huggingface/cache
export HF_DATASETS_CACHE=/path/to/datasets/cache
export DATA_DIR=/path/to/local/data

pytest tests/cli/test_smoke.py -v
```

If the data is already cached, `prepare_data` completes instantly.

## Markers

Tests use the following `pytest` markers (defined in `pyproject.toml`):

| Marker | Purpose                                          |
|--------|--------------------------------------------------|
| `slow` | Expensive tests. Deselect with `-m "not slow"`.  |
| `gpu`  | Tests that require a CUDA GPU.                   |
| `cli`  | Tests that invoke the `xlm` CLI as a subprocess. |

Combine markers:

```bash
pytest -m "cli and not slow"     # only fast CLI tests (e.g. scaffold --help)
pytest -m "cli and slow"         # only smoke tests
```

## How smoke tests work

Each smoke test invokes the `xlm` CLI as a subprocess:

```
xlm --config-dir tests/configs \
    job_type=train experiment=<experiment> debug=smoke \
    trainer_strategy=cpu paths.log_dir=<tmp_dir>
```

- **`--config-dir tests/configs`** prepends `tests/configs/` to
  Hydra's search path so that `debug=smoke` is discoverable without
  touching the production config directory.
- **`debug=smoke`** loads `tests/configs/debug/smoke.yaml`, which
  configures a minimal run: 5 training steps, validation at step 3
  (with `check_val_every_n_epoch: null` to override experiment-level
  epoch-based validation), batch size 1, 0 dataloader workers, no
  checkpointing.
- **`trainer_strategy=cpu`** forces CPU execution via the existing
  `trainer_strategy/cpu.yaml` config.

Success criteria: the subprocess exits with return code 0.

## Adding a new smoke test

Append one tuple to the `SMOKE_RUNS` list in `tests/cli/test_smoke.py`:

```python
SMOKE_RUNS: list[tuple[str, str]] = [
    ("star_easy_mlm",  "train"),
    ("star_easy_arlm", "train"),
    ("star_easy_ilm",  "train"),
    ("my_new_experiment", "train"),   # <-- add here
]
```

That is all.  The parametrized `test_smoke_run` function, the
`prepare_data` fixture, and the `smoke.yaml` config handle everything
else.  The new test will appear automatically as
`test_smoke_run[my_new_experiment-train]`.
