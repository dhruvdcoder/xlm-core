# Contributing to XLM

We value community contributions of all kinds, including code, documentation, bug reports, feature ideas, and helping others in [Discussions](https://github.com/dhruvdcoder/xlm-core/discussions).



## Ways to contribute

Pick the guide that matches your change:

| Type | Guide |
|------|--------|
| Add a maintained model (`xlm-models/`) | [Adding a model](https://dhruveshp.com/xlm-core/dev/guide/contributing/adding-a-model/) |
| Add an external model (separate repo) | [External models](https://dhruveshp.com/xlm-core/dev/guide/external-models/) |
| Add a task or dataset | [Adding a task or dataset](https://dhruveshp.com/xlm-core/dev/guide/adding-a-task/) |
| Core framework (`src/xlm/`) | [Core development](https://dhruveshp.com/xlm-core/dev/guide/contributing/core-development/) |
| Docs, bugs, issues | [Docs and issues](https://dhruveshp.com/xlm-core/dev/guide/contributing/docs-and-issues/) |

Overview and links: [Ways to contribute](https://dhruveshp.com/xlm-core/dev/guide/contributing/overview/).

## Getting started

The usual path is to **open an issue** (bug, feature, or design question) and then **open a pull request** that references it. Use the [pull request template](.github/PULL_REQUEST_TEMPLATE.md) when opening a PR.

### Good first issues

See [Good first issue](https://github.com/dhruvdcoder/xlm-core/issues?q=state%3Aopen+label%3A%22good+first+issue%22).

### Reporting bugs and features

- Search existing [Issues](https://github.com/dhruvdcoder/xlm-core/issues) first.
- Unsure if it is a bug or your setup? Ask in [Q&A Discussions](https://github.com/dhruvdcoder/xlm-core/discussions/categories/q-a).
- Use the issue templates: [bug report](.github/ISSUE_TEMPLATE/bug_report.md), [feature request](.github/ISSUE_TEMPLATE/feature_request.md), [new model](.github/ISSUE_TEMPLATE/new_model.md), [new task](.github/ISSUE_TEMPLATE/new_task.md).

## Environment setup

### Install from source

**Prerequisites:** Python 3.11+ (see [`setup.py`](setup.py)) and [conda](https://docs.conda.io/en/latest/).

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

3. Create and activate a **virtual environment**:

   ```bash
   conda create -p .venv_xlm_core python=3.11.10 pip ipykernel -y
   conda activate ./.venv_xlm_core
   ```

4. Install **xlm-core** and **xlm-models** in editable mode (same optional extras as CI):

   ```bash
   pip install -e ".[all]"
   pip install -e ./xlm-models
   ```

5. Install **development tooling** (test, dev, docs, lint):

   ```bash
   pip install -r requirements/test_requirements.txt \
               -r requirements/dev_requirements.txt \
               -r requirements/docs_requirements.txt \
               -r requirements/lint_requirements.txt
   ```

6. **Verify** the installation:

   ```bash
   xlm --help
   pytest -m "not slow and not cli"
   ```

If you only need a subset of tooling (for example docs-only), see [Dependencies](https://dhruveshp.com/xlm-core/dev/developers/dependencies/) for what each requirements file contains.

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

**Coverage** (configuration in `pyproject.toml`):

```bash
coverage run -m pytest -m "not slow and not cli"
coverage report
```

CI runs `pytest -m "not gpu and not cli and not integration"` on pull requests (see [`.github/workflows/tests.yml`](.github/workflows/tests.yml)).

## Style and static checks

There is no repo-wide Makefile; run tools from the activated environment.

- **Format:** `black` is configured in `pyproject.toml` (line length 79):

  ```bash
  black src xlm-models tests
  ```

- **Lint / types:** the lint requirements bundle includes `flake8`, `mypy`, and related plugins. Fix new warnings in code you change; avoid drive-by mass reformatting of unrelated files.

## Documentation

- Sources live under [`docs/`](docs/). Build locally with `mkdocs build` (see [Docs and issues](https://dhruveshp.com/xlm-core/dev/guide/contributing/docs-and-issues/)).
- Add new pages to the `nav:` in [`mkdocs.yml`](mkdocs.yml) when they should appear in the sidebar.

## Opening a pull request

1. Push your branch to **your fork** and open a PR against **`main`** on [dhruvdcoder/xlm-core](https://github.com/dhruvdcoder/xlm-core).
2. Fill out the [pull request template](.github/PULL_REQUEST_TEMPLATE.md): contribution type, summary, testing, docs.
3. Link related issues and note any new dependencies or optional extras.
4. Ensure **tests** and **docs** you touched still build.
5. Request review from maintainers when ready.

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project ([MIT](LICENSE)).
