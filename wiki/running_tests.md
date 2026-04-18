# Running Tests

## Prerequisites

1. **Install test dependencies** (if not already installed):

   ```bash
   pip install -r requirements/test_requirements.txt
   ```

   This installs `pytest`, `hypothesis`, and `coverage[toml]`.

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
    │                        # see wiki/integration_tests.md for the architecture.
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

## End-to-end CLI smoke tests

These tests live in [`tests/cli/test_smoke.py`](tests/cli/test_smoke.py). Each case runs the real `xlm` console script as a **subprocess** with:

- `--config-dir` pointing at [`tests/configs`](tests/configs), so Hydra can resolve the `debug` group.
- `debug=smoke`, which loads [`tests/configs/debug/smoke.yaml`](tests/configs/debug/smoke.yaml): a minimal run (5 training steps, validation at step 3, batch size 1, CPU strategy, no checkpointing).
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

**Add a new combo:** append a `(experiment_name, job_type)` tuple to `SMOKE_RUNS` in [`tests/cli/test_smoke.py`](tests/cli/test_smoke.py). The parametrized test id becomes `experiment_name-job_type`.

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
`lightning.Trainer` runs.  See [`integration_tests.md`](./integration_tests.md)
for the full architecture, fixtures, and a mermaid diagram.

> **Looking for the per-method `DatasetManager` tests?**  Every branch
> of `__init__` / `prepare_data` / `setup` / `get_dataloader` is
> covered by `tests/core/test_datamodule.py` (single-process, ~1 s,
> runs as part of the fast unit suite).  The integration suite
> covers what those tests *cannot* exercise in a single process.

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

Spawns Python subprocesses with `gloo`.  Expect ~10-15 s per test.
Requires nothing beyond the standard test environment.

### Run the SLURM tier (multi-GPU)

The SLURM tests are double-gated: they auto-skip without `sbatch` on
PATH and *also* without `XLM_INTEGRATION_SLURM_ENABLE=1`.  This
prevents accidental job submissions on cluster login nodes.

```bash
XLM_INTEGRATION_SLURM_ENABLE=1 \
pytest -m "integration and slurm" tests/integration/
```

If the default partition / GRES in
[`tests/integration/datamodule/slurm/ddp_iterable_shards/script.sh`](../tests/integration/datamodule/slurm/ddp_iterable_shards/script.sh)
does not match your cluster, override via
`XLM_INTEGRATION_SBATCH_ARGS` (comma-separated, forwarded verbatim
to `sbatch`):

```bash
XLM_INTEGRATION_SLURM_ENABLE=1 \
XLM_INTEGRATION_SBATCH_ARGS="--partition=gpu,--qos=debug" \
pytest -m "integration and slurm" tests/integration/
```

Per-rank `rank_<RANK>.json` result files and SLURM `slurm-*.out` logs
land in pytest's `tmp_path`; on failure the assertion message points
at the directory.

### Run the entire integration suite locally

```bash
pytest -m "integration and not slurm" tests/integration/
```

This runs everything that does not require SLURM (~2 min).

## Writing new tests

- **Shared fixtures** live in `tests/conftest.py` (root) or in the
  relevant sub-package's `conftest.py`. Use `simple_tokenizer`,
  `dummy_noise_schedule`, `tiny_model_kwargs`, etc. rather than
  creating your own.
- **Model tests** follow a 4-file pattern per model family:
  `test_model_*.py`, `test_loss_*.py`, `test_predictor_*.py`,
  `test_collator_*.py`.
- **Mark expensive tests** with `@pytest.mark.slow` so they are excluded
  from the fast development loop.
- **Use `pytest.skip()`** with a clear message when a test requires
  infrastructure that is not yet available (e.g. a tiny experiment config).

## Test mixin pattern for model families

To avoid duplicating the same test logic across every model family we use
**base test mixin classes** defined in `tests/models/_base.py`.

### Available mixins

| Mixin              | Shared tests                                                                                      | Required fixtures                       |
|--------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------|
| `BaseModelTests`   | `test_forward_output_shape`, `test_forward_with_partial_mask`, `test_gradient_flows`, `test_weight_decay_param_split` | `model`, `run_forward`                  |
| `BaseLossTests`    | `test_returns_loss_key`, `test_loss_is_scalar`, `test_loss_is_finite`, `test_loss_requires_grad`, `test_gradients_reach_model` | `loss_fn`, `batch`                      |
| `BaseCollatorTests`| `test_output_has_target_ids`, `test_output_shapes_consistent`                                     | `collator`, `raw_examples`              |

### How it works

Each concrete test class inherits from the appropriate mixin and
provides a small set of **adapter fixtures** that wire the generic
tests to the specific model under test. The mixin contributes the test
methods; the subclass contributes the fixtures.

```
 ┌───────────────────────────┐
 │   BaseModelTests (mixin)  │  ← generic test methods
 │   - test_forward_output_…│
 │   - test_gradient_flows   │
 │   - ...                   │
 └───────────┬───────────────┘
             │ inherits
 ┌───────────▼───────────────┐
 │ TestRotaryTransformerMLM…  │  ← only provides fixtures
 │   + model fixture          │
 │   + run_forward fixture    │
 │   + (optional extra tests) │
 └────────────────────────────┘
```

### Key fixture: `run_forward`

The `run_forward` fixture is a **callable** with the signature:

```python
def run_forward(batch_size=2, seq_len=16, partial_mask=False) -> Tensor
```

It builds the appropriate input tensors for that model family, calls
`model.forward()`, and returns the **vocab-logits tensor** (shape
`(batch, seq, vocab)`). This single fixture encapsulates all the
differences between model forward signatures:

- **MLM** passes `(x, attention_mask=..., positions=...)`
- **ARLM** builds a 3-D causal mask
- **MDLM** additionally passes a noise-time `t` tensor
- **ILM** unpacks the `(vocab_logits, length_logits)` tuple

### Adding tests for a new model family

1. **Create the directory** `tests/models/<family>/` with `__init__.py`
   and `conftest.py`.
2. **Add a model fixture** to `tests/models/conftest.py`:

   ```python
   @pytest.fixture()
   def tiny_foo_model(tiny_model_kwargs):
       from foo.model_foo import FooModel
       return FooModel(**tiny_model_kwargs)
   ```

3. **Create `test_model_foo.py`** – inherit from `BaseModelTests` and
   provide `model` + `run_forward`:

   ```python
   from tests.models._base import BaseModelTests

   class TestFooModel(BaseModelTests):
       @pytest.fixture()
       def model(self, tiny_foo_model):
           return tiny_foo_model

       @pytest.fixture()
       def run_forward(self, model, simple_tokenizer):
           def _run(batch_size=2, seq_len=16, partial_mask=False):
               x = torch.randint(0, simple_tokenizer.vocab_size, (batch_size, seq_len))
               mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
               if partial_mask:
                   mask[:, -4:] = False
               positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
               return model(x, attention_mask=mask, positions=positions)
           return _run

       # Add any model-specific tests as extra methods here.
   ```

4. **Create `test_loss_foo.py`** – inherit from `BaseLossTests`:

   ```python
   from tests.models._base import BaseLossTests

   class TestFooLoss(BaseLossTests):
       @pytest.fixture()
       def loss_fn(self, tiny_foo_model, simple_tokenizer):
           return FooLoss(model=tiny_foo_model, tokenizer=simple_tokenizer)

       @pytest.fixture()
       def batch(self, foo_batch):
           return foo_batch
   ```

5. **Create `test_collator_foo.py`** – inherit from `BaseCollatorTests`:

   ```python
   from tests.models._base import BaseCollatorTests

   class TestFooCollator(BaseCollatorTests):
       @pytest.fixture()
       def collator(self, simple_tokenizer, dummy_noise_schedule):
           return FooCollator(tokenizer=simple_tokenizer, block_size=32,
                              noise_schedule=dummy_noise_schedule)

       @pytest.fixture()
       def raw_examples(self, simple_tokenizer):
           return [{"input_ids": [...], "attention_mask": [...], ...} for _ in range(4)]
   ```

6. Run `pytest tests/models/<family>/ -v` and verify all inherited +
   custom tests pass.

### Extending a mixin with model-specific tests

Simply add extra test methods to the concrete class. They coexist
with the inherited ones:

```python
class TestRotaryTransformerILMModel(BaseModelTests):
    # ... fixtures ...

    # ILM-specific test not in the base mixin
    def test_length_logits_is_none(self, model, simple_tokenizer):
        ...
```
