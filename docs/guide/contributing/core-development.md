# Core framework development

Changes to **xlm-core** live under {{ gh_dir('src/xlm', 'src/xlm/') }} — the training harness, datamodule, metrics, shared modules, tasks, Hydra configs, and CLI. This is separate from model families in {{ gh_dir('xlm-models', 'xlm-models/') }}.



## What counts as "core"

| Area | Location | Examples |
|------|----------|----------|
| Training harness | {{ gh('src/xlm/harness.py', 'harness.py') }} | `Harness`, `LossFunction`, `Predictor` protocols |
| Data pipeline | {{ gh('src/xlm/datamodule.py', 'datamodule.py') }} | `TextDataModule`, `DatasetManager`, collator protocols |
| Metrics | {{ gh('src/xlm/metrics.py', 'metrics.py') }} | `MetricWrapper`, custom metrics |
| Shared modules | {{ gh_dir('src/xlm/modules', 'modules/') }} | Rotary transformer, DDiT backbones |
| Tasks / datasets | {{ gh_dir('src/xlm/tasks', 'tasks/') }} | Preprocessing, post-hoc evaluators |
| CLI | {{ gh_dir('src/xlm/commands', 'commands/') }} | `xlm`, `xlm-scaffold`, job dispatch |
| Hydra configs | {{ gh_dir('src/xlm/configs', 'configs/') }} | Shared dataset YAMLs, trainer defaults |

## Prerequisites

Set up your environment first: [Install from source](https://github.com/dhruvdcoder/xlm-core/blob/main/CONTRIBUTING.md#install-from-source) in the Contributing hub.

For core work, install the same stacks CI uses so optional task paths and model families stay importable:

```bash
pip install -e ".[all]"
pip install -e ./xlm-models
pip install -r requirements/test_requirements.txt \
            -r requirements/dev_requirements.txt \
            -r requirements/docs_requirements.txt \
            -r requirements/lint_requirements.txt
```

Protocol or harness changes can break any model family — keep `xlm-models` editable while you develop.

## Before you start

1. **Search existing [Issues](https://github.com/dhruvdcoder/xlm-core/issues)** and [Discussions](https://github.com/dhruvdcoder/xlm-core/discussions) for overlap.
2. **Open an issue** for non-trivial work. Use the [feature request template](https://github.com/dhruvdcoder/xlm-core/issues/new?template=feature_request.md) and note which component is affected (core, CLI, metrics, datamodule, etc.).
3. **Assess impact radius:**
   - **Protocol changes** (`LossFunction`, `Predictor`, `Collator`, or `Harness` in {{ gh('src/xlm/harness.py', 'harness.py') }} / {{ gh('src/xlm/datamodule.py', 'datamodule.py') }}) affect every family under {{ gh_dir('xlm-models', 'xlm-models/') }} (`arlm`, `ilm`, `mlm`, `mdlm`, `flexmdm`, `dream`) and external models using the same contracts.
   - **Hydra config changes** (renamed keys, moved defaults under {{ gh_dir('src/xlm/configs', 'configs/') }}) may require updating experiment and datamodule YAMLs across `xlm-models/` and downstream repos.
4. Skim the closest existing code and the [Related guides](#related-guides) below before proposing API shape.

## Implementation conventions

Follow repo conventions in the files you touch:

- **Types:** use `jaxtyping` for tensor shapes where applicable (same pattern as model families).
- **Docstrings:** Google style (see {{ gh('.darglint', '.darglint') }} and {{ gh('.flake8', '.flake8') }}).
- **Formatting:** line length 79 — `black src xlm-models tests` (see {{ gh('pyproject.toml', 'pyproject.toml') }}).
- **Protocols:** if you add or change a protocol in `harness.py` or `datamodule.py`, update docstrings and consider backward compatibility for existing model packages.
- **Configs:** if you add Hydra groups or change shared YAMLs, update affected experiment configs and mention the migration in your PR.

There is no single “core feature” template — use neighboring modules and tests as reference.

## Related guides

Depending on what you are changing, these existing pages may help:

- [Data pipeline](../data-pipeline.md) — how `DatasetManager` and dataloader names connect to metrics
- [Metrics](../metrics.md) — defining and wiring metrics
- [Custom commands](../custom-commands.md) — extending `job_type` via external model `commands.yaml`
- [Adding a task or dataset](../adding-a-task.md) — if your core change includes a new benchmark
- [Debugging utilities](../debugging-utils.md) — debug modes and utilities

## Testing

- **Unit tests:** add or extend tests under {{ gh_dir('tests/core', 'tests/core/') }}.
- **Datamodule / collator changes:** update matrix coverage in {{ gh('tests/core/test_datamodule.py', 'test_datamodule.py') }}; consider scenarios under {{ gh_dir('tests/integration', 'tests/integration/') }} — see [Integration tests](../../developers/testing/integration-tests.md).
- **Harness or protocol changes:** run model-family tests to catch regressions:

  ```bash
  pytest tests/models/ -v
  ```

- **Fast loop (while developing):**

  ```bash
  pytest -m "not slow and not cli"
  ```

- **CI-aligned run** (matches {{ gh('.github/workflows/tests.yml', 'tests.yml') }}):

  ```bash
  pytest -m "not gpu and not cli and not integration"
  ```

- **Style:** format and lint changed paths, e.g. `black src xlm-models tests` and `flake8 src/xlm/` on files you modified.

See [Running tests](../../developers/testing/running-tests.md) for markers, coverage, and smoke tests.

## Documentation

- **Public API:** modules under {{ gh_dir('src/xlm', 'src/xlm/') }} are largely covered by `api-autonav` for `src/xlm` in {{ gh('mkdocs.yml', 'mkdocs.yml') }}; add docstrings on new public symbols.
- **Conceptual changes:** if you introduce a new user-facing workflow, consider a page under {{ gh_dir('docs/guide', 'docs/guide/') }} and a `nav:` entry in `mkdocs.yml`.
- **Verify the site builds:**

  ```bash
  mkdocs build
  ```

More detail: [Docs and issues](docs-and-issues.md).

## Opening a PR

1. Push your branch to your fork and open a PR against `main`.
2. Use the [pull request template](https://github.com/dhruvdcoder/xlm-core/blob/main/.github/PULL_REQUEST_TEMPLATE.md) and check **Core framework (`src/xlm/`)** under contribution type.
3. Summarize what behavior or protocol changed and **why**.
4. Call out **breaking changes** for model families or external packages.
5. Link the related issue and note tests/docs you ran or updated.

General workflow: [Contributing hub](https://github.com/dhruvdcoder/xlm-core/blob/main/CONTRIBUTING.md#opening-a-pull-request).
