# Dependencies

This page describes how dependencies are laid out for **xlm-core** and how to install subsets for runtime, testing, docs, linting, and extras.

## How packages are wired

[`setup.py`](https://github.com/dhruvdcoder/xlm-core/blob/main/setup.py) loads **`requirements.txt`** at the repo root via `install_requires`:

- `pip install xlm-core` installs those runtime dependencies.
- **`requirements/core_requirements.txt`** is a complementary file with stricter pins, extra comments (e.g. Lightning / Torch / datasets), and tooling lines used for local/full dev installs—not every line is mirrored in `requirements.txt`.

Run commands below from the **repository root** (where `setup.py` and `requirements/` live).

## Requirement files overview

| File | Purpose |
|------|---------|
| [`requirements.txt`](https://github.com/dhruvdcoder/xlm-core/blob/main/requirements.txt) (root) | Runtime dependencies used when publishing / `pip install xlm-core` |
| [`requirements/core_requirements.txt`](https://github.com/dhruvdcoder/xlm-core/blob/main/requirements/core_requirements.txt) | Full core stack with version pins and inline notes |
| [`requirements/test_requirements.txt`](https://github.com/dhruvdcoder/xlm-core/blob/main/requirements/test_requirements.txt) | pytest, hypothesis, coverage |
| [`requirements/dev_requirements.txt`](https://github.com/dhruvdcoder/xlm-core/blob/main/requirements/dev_requirements.txt) | Notebooks / experiment tracking helpers |
| [`requirements/docs_requirements.txt`](https://github.com/dhruvdcoder/xlm-core/blob/main/requirements/docs_requirements.txt) | MkDocs site build |
| [`requirements/lint_requirements.txt`](https://github.com/dhruvdcoder/xlm-core/blob/main/requirements/lint_requirements.txt) | Typecheck, formatting, flake8 ecosystem |
| [`requirements/extra_requirements.txt`](https://github.com/dhruvdcoder/xlm-core/blob/main/requirements/extra_requirements.txt) | Optional analysis / plotting (lighter set) |
| [`requirements/plotting_requirements.txt`](https://github.com/dhruvdcoder/xlm-core/blob/main/requirements/plotting_requirements.txt) | Heavier plotting / stats stack |

---

## Root `requirements.txt`

**Install:**

```bash
pip install -r requirements.txt
```

**Role:** Canonical list consumed by setuptools for `pip install xlm-core`. Matches the pinned core stack where it matters but may omit duplicated or heavily commented lines present in `core_requirements.txt`.

---

## Core requirements (`requirements/core_requirements.txt`)

**Install:**

```bash
pip install -r requirements/core_requirements.txt
```

**Role:** Day-to-day / CI-style full environment for training and pipelines: PyTorch, Lightning, Hydra extras, HF `datasets`, `transformers`, `torchdata`, `jaxtyping`, `tensorboard`, `simple_slurm`, etc.

Inline comments document choices (Torch range, Lightning version, datasets upper bound).

---

## Test dependencies

**Install:**

```bash
pip install -r requirements/test_requirements.txt
```

**Included packages:**

- `pytest`
- `hypothesis`
- `coverage[toml]`

For workflows and invocation, see also [Running tests](testing/running-tests.md).

Coverage configuration and pytest markers live in `pyproject.toml`.

---

## Development (`requirements/dev_requirements.txt`)

**Install:**

```bash
pip install -r requirements/dev_requirements.txt
```

Includes `jupytext` and `wandb`.

---

## Documentation build (`requirements/docs_requirements.txt`)

**Install:**

```bash
pip install -r requirements/docs_requirements.txt
```

Includes `mkdocs`, `mkdocs-material`, `mkdocstrings` (Python handler), `mike`, and `mkdocs-api-autonav`. Used when building this site locally or in CI.

---

## Lint / quality (`requirements/lint_requirements.txt`)

**Install:**

```bash
pip install -r requirements/lint_requirements.txt
```

Includes `mypy`, `pre-commit`, `black`, `flake8` and related plugins (`flake8-docstrings`, `flake8-annotations`, `flake8-black`), `pytest-flake8`, `autoflake`, `darglint`, and `pre-commit-hooks`.

---

## Extra (`requirements/extra_requirements.txt`)

**Install:**

```bash
pip install -r requirements/extra_requirements.txt
```

Includes `matplotlib`, `seaborn`, `networkx`, and `wandb` (overlapping with dev for experiment tracking).

---

## Plotting (`requirements/plotting_requirements.txt`)

**Install:**

```bash
pip install -r requirements/plotting_requirements.txt
```

Includes `matplotlib`, `seaborn`, `statsmodels`, and `tueplots` for richer plotting notebooks or papers-style figures.
