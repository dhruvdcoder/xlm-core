# Releasing to PyPI

This page describes how to publish **xlm-core** to PyPI and refresh release docs.

## Automated release (recommended)

Use [`.github/release.py`](https://github.com/dhruvdcoder/xlm-core/blob/main/.github/release.py). It:

1. Updates default values in `src/xlm/version.py`
2. Verifies `VERSION` matches the requested release
3. Commits and pushes to `main`
4. Runs `gh release create v<version>` (triggers PyPI upload and docs deploy)

The GitHub release **tag** is always `v` + the version in `version.py` after the bump, so the tag and file stay aligned.

### Prerequisites

- [`gh`](https://cli.github.com/) installed and authenticated (`gh auth login`)
- Push access to `main` on [dhruvdcoder/xlm-core](https://github.com/dhruvdcoder/xlm-core)
- Clean working tree (no uncommitted changes)

### Local

```bash
cd xlm-core
python .github/release.py 0.1.4          # interactive confirm
python .github/release.py 0.1.4 --yes    # non-interactive
python .github/release.py 0.1.4 --dry-run
```

Pre-releases use a hyphen suffix (e.g. `0.1.4-alpha` → tag `v0.1.4-alpha`).

If the version bump is already on `main` but the release was not created:

```bash
python .github/release.py --publish-only --yes
```

### GitHub Actions

**Actions → Release xlm-core → Run workflow**, enter the version (and optionally enable dry run).

Requires `contents: write` and `actions: write` (configured in [`.github/workflows/release.yml`](https://github.com/dhruvdcoder/xlm-core/blob/main/.github/workflows/release.yml)).

After the release is created, that workflow **explicitly triggers** [publish.yml](https://github.com/dhruvdcoder/xlm-core/blob/main/.github/workflows/publish.yml) and [docs-release.yml](https://github.com/dhruvdcoder/xlm-core/blob/main/.github/workflows/docs-release.yml) via `workflow_dispatch`. GitHub does not run those workflows automatically when the release is created with `GITHUB_TOKEN` (only when you publish from the UI or use `gh` with a personal token locally).

To publish a release that was already created but missed PyPI/docs:

**Actions → Upload Python Package → Run workflow** with `tag_name` = e.g. `v0.1.4` (same for **Deploy Release Docs to GitHub Pages**).

## What happens after the release

| Workflow | Trigger | Effect |
|----------|---------|--------|
| [publish.yml](https://github.com/dhruvdcoder/xlm-core/blob/main/.github/workflows/publish.yml) | `release: published` | Build wheel/sdist; upload to PyPI |
| [docs-release.yml](https://github.com/dhruvdcoder/xlm-core/blob/main/.github/workflows/docs-release.yml) | `release: published` | Deploy versioned docs via mike |

PyPI build version comes from the **release tag** (`XLM_CORE_VERSION_*` env vars override `version.py` at build time). Keeping `version.py` in sync avoids confusion on `main` after install.

## Required secrets

Repo **Settings → Secrets and variables → Actions**:

- `PYPI_USERNAME` – PyPI username
- `PYPI_TOKEN` – PyPI API token

## Manual release (legacy)

1. Update defaults in `src/xlm/version.py` (e.g. `0.1.2`).
2. Commit and push to `main`.
3. Create a GitHub release with tag `v0.1.2` (or `0.1.2`) on that commit and publish it.

Root [`setup.py`](https://github.com/dhruvdcoder/xlm-core/blob/main/setup.py) reads `VERSION` from `version.py` via `exec()`.

## xlm-models

[`xlm-models/setup.py`](https://github.com/dhruvdcoder/xlm-core/blob/main/xlm-models/setup.py) is a **separate** package with its own hardcoded version. It is **not** published by `publish.yml`. Bump it manually if you publish `xlm-models` separately.

## Version format

Supported: `MAJOR.MINOR.PATCH` with optional pre-release suffix `-label` (e.g. `0.1.4-alpha`). Tags use a `v` prefix (`v0.1.4`).
