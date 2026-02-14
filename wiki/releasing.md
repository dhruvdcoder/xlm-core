# Releasing to PyPI

This page describes how to publish xlm-core to PyPI.

## Pre-release Step

Update the version in `src/xlm/version.py`. The `setup.py` reads `VERSION` from this file via `exec()`. The publish workflow sets `XLM_CORE_VERSION_*` environment variables from the GitHub release tag, overriding `version.py` at build time. For consistency and local development (`import xlm; xlm.__version__`), update `version.py` (defaults or hardcoded values) to match the release version before creating the release.

## Release Workflow

The [.github/workflows/publish.yml](https://github.com/dhruvdcoder/xlm-core/blob/main/.github/workflows/publish.yml) workflow triggers on **GitHub release creation** (`release: types: [published]`).

## Process

1. Update `src/xlm/version.py` to match the version you will tag (e.g., `0.1.2`).
2. Commit and push the version change.
3. Create a new release on GitHub (Releases → Draft a new release).
4. Create a tag matching the version (e.g., `v0.1.2` or `0.1.2`).
5. Publish the release. The workflow extracts the version from the tag, builds the package, and uploads to PyPI via Twine.

## Required Secrets

Configure these in the repo Settings → Secrets and variables → Actions:

- `PYPI_USERNAME` – PyPI username
- `PYPI_TOKEN` – PyPI API token

## How It Works

The workflow parses the release tag (stripping the `v` prefix) and passes `XLM_CORE_VERSION_MAJOR`, `XLM_CORE_VERSION_MINOR`, `XLM_CORE_VERSION_PATCH`, and `XLM_CORE_VERSION_SUFFIX` as environment variables to override `version.py` at build time.
