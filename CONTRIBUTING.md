# Contributing to XLM

We value community contributions of all kinds, including code, documentation, bug reports, feature ideas, and helping others in [Discussions](https://github.com/dhruvdcoder/xlm-core/discussions).

This guide was informed by the [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md) and [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/contributing) contributing guides.

**Docs site (dev):** [https://dhruveshp.com/xlm-core/dev/](https://dhruveshp.com/xlm-core/dev/)

## Quick links

| Topic | Where |
|-------|--------|
| Dependencies and requirement files | [Dependencies](https://dhruveshp.com/xlm-core/dev/developers/dependencies/) |
| Running tests | [Running tests](https://dhruveshp.com/xlm-core/dev/developers/testing/running-tests/) |
| Unit / integration testing | [Unit tests](https://dhruveshp.com/xlm-core/dev/developers/testing/unit-tests/), [Integration tests](https://dhruveshp.com/xlm-core/dev/developers/testing/integration-tests/) |
| Model families (shared abstractions) | [Models overview](https://dhruveshp.com/xlm-core/dev/models/) |
| External / fork models | [External models](https://dhruveshp.com/xlm-core/dev/guide/external-models/) |
| New task or dataset | [Adding a task or dataset](https://dhruveshp.com/xlm-core/dev/guide/adding-a-task/) |

## Contributing code or documentation

The usual path is to **open an issue** (bug, feature, or design question) and then **open a pull request** that references it.

### Good first issues

If you want a small starter task, see [Good first issue](https://github.com/dhruvdcoder/xlm-core/issues?q=state%3Aopen+label%3A%22good+first+issue%22).

### Reporting bugs

- Search existing [Issues](https://github.com/dhruvdcoder/xlm-core/issues) first.
- If you are not sure whether something is a library bug or your setup, ask in [Q&A Discussions](https://github.com/dhruvdcoder/xlm-core/discussions/categories/q-a).

### Feature requests

Open an issue and use the feature request template when available.

## Environment setup

You need **Python 3.11+** (see [`setup.py`](https://github.com/dhruvdcoder/xlm-core/blob/main/setup.py)).

1. **Fork** the [repository](https://github.com/dhruvdcoder/xlm-core) and **clone your fork**:

   ```bash
   git clone git@github.com:<your-github-username>/xlm-core.git
   cd xlm-core
   git remote add upstream https://github.com/dhruvdcoder/xlm-core.git
   ```

2. Create a **branch** (do not commit directly on `main`):

   ```bash
   git checkout -b short-description-of-change
   ```

3. Create a virtual environment and install **xlm-core** in editable mode, plus dev / test / docs / lint stacks:

   ```bash
   pip install -e .
   pip install -r requirements/dev_requirements.txt
   pip install -r requirements/test_requirements.txt
   pip install -r requirements/docs_requirements.txt
   pip install -r requirements/lint_requirements.txt
   ```

   Optional: install the same extras as CI when you need optional task stacks:

   ```bash
   pip install -e ".[all]"
   ```

4. If you change **model code** under `xlm-models/`, install that package in editable mode as well (it is a separate setuptools package in the same repo):

   ```bash
   pip install -e ./xlm-models
   ```

   `pip install -e .` alone does not install `xlm-models` unless you already installed it from PyPI.

## Running tests

Details: [Running tests](https://dhruveshp.com/xlm-core/dev/developers/testing/running-tests/).

**Fast loop (recommended while developing):**

```bash
pytest -m "not slow and not cli"
```

**Full suite** (includes slow and CLI subprocess tests; some tests skip if resources are missing):

```bash
pytest
```

**Focused runs:**

```bash
pytest tests/core/
pytest tests/models/mlm/
pytest tests/cli/test_smoke.py -m "cli and slow" -v
```

**Coverage** (configuration in `pyproject.toml`):

```bash
coverage run -m pytest -m "not slow and not cli"
coverage report
```

## Style and static checks

There is no repo-wide Makefile; run tools from the activated environment.

- **Format:** `black` is configured in `pyproject.toml` (line length 79). Example:

  ```bash
  black src xlm-models tests
  ```

- **Lint / types:** the lint requirements bundle includes `flake8`, `mypy`, and related plugins. Run them the same way you would in other Python projects (e.g. `flake8 src xlm-models tests`, `mypy` with your usual targets). Align with what CI enforces for the paths you touch.

Fix new warnings in code you change; avoid drive-by mass reformatting of unrelated files.

## Documentation

- Sources live under [`docs/`](https://github.com/dhruvdcoder/xlm-core/tree/main/docs). The published site is built with MkDocs (see [`mkdocs.yml`](https://github.com/dhruvdcoder/xlm-core/blob/main/mkdocs.yml)).
- **Local build:**

  ```bash
  mkdocs build
  ```

- Add new pages to the `nav:` in `mkdocs.yml` when they should appear in the sidebar.

## Contributing a new model

We ship reference implementations in the **`xlm-models`** package in this repository (sibling of `src/xlm/`).

### External model (separate repo)

For maximum flexibility (e.g. reproducing a paper with its own layout), use the **external models** mechanism. See the [External models guide](https://dhruveshp.com/xlm-core/dev/guide/external-models/).

### Model inside `xlm-models`

To add a **first-party** family under `xlm-models/<name>/`, follow the same component pattern as existing families:

| Component | Role |
|-----------|------|
| **Model** | Architecture and forward pass |
| **Loss** | Training objective |
| **Predictor** | Inference / generation |
| **Collator** | Batching |

Conceptual comparison and batch contracts: [Models overview](https://dhruveshp.com/xlm-core/dev/models/). Per-family pages (ARLM, ILM, MDLM, MLM) show the expected module layout.

**Checklist:**

1. Implement the family under `xlm-models/<family>/` (mirror `model_*`, `loss_*`, `predictor_*`, `datamodule_*`, `types_*`, `metrics_*`, etc., as appropriate).
2. Add Hydra configs under `xlm-models/<family>/configs/` (datamodule, collator, experiment, …).
3. Register the family in [`xlm-models/xlm_models.json`](https://github.com/dhruvdcoder/xlm-core/blob/main/xlm-models/xlm_models.json) if it should be discovered like existing tags.
4. Add tests under `tests/models/<family>/` using the shared mixins in [`tests/models/_base.py`](https://github.com/dhruvdcoder/xlm-core/blob/main/tests/models/_base.py). See [Unit tests](https://dhruveshp.com/xlm-core/dev/developers/testing/unit-tests/).
5. Extend **MkDocs** if you want a narrative family page: add `docs/models/<family>.md` and a nav entry in `mkdocs.yml`, and add the module path to `api-autonav` in `mkdocs.yml` if you want it in the API Reference section.
6. Consider a CLI smoke entry in `tests/cli/test_smoke.py` once a minimal experiment config exists.

## Contributing a new task or dataset

Wire preprocessing in `src/xlm/tasks/<your_task>/`, add dataset YAMLs under `src/xlm/configs/lightning_train/datasets/`, and align dataloader names with metrics and evaluators. Step-by-step: [Adding a task or dataset](https://dhruveshp.com/xlm-core/dev/guide/adding-a-task/), with background in the [Data pipeline](https://dhruveshp.com/xlm-core/dev/guide/data-pipeline/) and [Metrics](https://dhruveshp.com/xlm-core/dev/guide/metrics/) guides.

## Opening a pull request

1. Push your branch to **your fork** and open a PR against **`main`** on [dhruvdcoder/xlm-core](https://github.com/dhruvdcoder/xlm-core).
2. Describe the change, link related issues, and note any new dependencies or optional extras.
3. Ensure **tests** and **docs** you touched still build.
4. Request review from maintainers (e.g. `dhruvdcoder`, `brozonoyer`, `sensai99`, `Durga-Prasad1`) when ready.

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project ([MIT](https://github.com/dhruvdcoder/xlm-core/blob/main/LICENSE)).
