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
├── conftest.py              # Shared fixtures (tokenizer, batches, tiny model kwargs)
├── core/                    # Unit tests for xlm core components
│   ├── test_tokenizers.py
│   ├── test_collators.py
│   ├── test_noise.py
│   ├── test_metrics.py
│   ├── test_datamodule.py
│   └── test_harness.py
├── models/                  # Unit tests for each model family
│   ├── _base.py             # Base test mixin classes (BaseModelTests, BaseLossTests, …)
│   ├── mlm/                 # model, loss, predictor, collator
│   ├── mdlm/
│   ├── arlm/
│   └── ilm/
└── cli/                     # CLI / integration tests
    ├── test_train.py
    ├── test_eval.py
    ├── test_generate.py
    └── test_scaffold.py
```

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
