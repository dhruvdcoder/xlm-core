# Dependencies

This page describes how dependencies are laid out for **xlm-core** and how to install subsets for runtime, testing, docs, linting, and extras.

## Contributor install from source

For a full contributor setup from a git checkout, follow [Install from source](https://github.com/dhruvdcoder/xlm-core/blob/main/CONTRIBUTING.md#install-from-source) in {{ gh('CONTRIBUTING.md', 'CONTRIBUTING.md') }}. That page has the canonical conda + pip recipe (aligned with CI).

Use the sections below to understand what each requirements file contains, or to install a subset if you only need part of the stack (for example docs-only).

## How packages are wired

{{ gh('setup.py', 'setup.py') }} loads **`requirements.txt`** at the repo root via `install_requires`:

- `pip install xlm-core` installs those runtime dependencies.
- **`requirements/core_requirements.txt`** is a complementary file with stricter pins, extra comments (e.g. Lightning / Torch / datasets), and tooling lines used for local/full dev installs—not every line is mirrored in `requirements.txt`.

Run commands below from the **repository root** (where `setup.py` and `requirements/` live).

## Requirement files overview

| File | Purpose |
|------|---------|
| {{ gh('requirements.txt', 'requirements.txt') }} (root) | Runtime dependencies used when publishing / `pip install xlm-core` |
| {{ gh('requirements/core_requirements.txt', 'requirements/core_requirements.txt') }} | Full core stack with version pins and inline notes |
| {{ gh('requirements/test_requirements.txt', 'requirements/test_requirements.txt') }} | pytest, hypothesis, coverage |
| {{ gh('requirements/dev_requirements.txt', 'requirements/dev_requirements.txt') }} | Notebooks / experiment tracking helpers |
| {{ gh('requirements/docs_requirements.txt', 'requirements/docs_requirements.txt') }} | MkDocs site build |
| {{ gh('requirements/lint_requirements.txt', 'requirements/lint_requirements.txt') }} | Typecheck, formatting, flake8 ecosystem |
| {{ gh('requirements/extra_requirements.txt', 'requirements/extra_requirements.txt') }} | Optional analysis / plotting (lighter set) |
| {{ gh('requirements/plotting_requirements.txt', 'requirements/plotting_requirements.txt') }} | Heavier plotting / stats stack |

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

Includes `mkdocs`, `mkdocs-material`, `mkdocstrings` (Python handler), `mike`, `mkdocs-api-autonav`, and `mkdocs-macros-plugin`. Used when building this site locally or in CI. Source links in docs use the `gh()` / `gh_dir()` macros defined in {{ gh('main.py', 'main.py') }} at the repo root.

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
