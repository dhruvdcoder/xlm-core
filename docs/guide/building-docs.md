# Building Docs

This page describes how the XLM documentation is built and deployed.

## Local Build

Install dependencies and build the docs:

```bash
pip install -e .
pip install -e xlm-models/
pip install -r requirements/docs_requirements.txt
mkdocs build
```

## Local Serve

To preview the docs with live reload:

```bash
mkdocs serve
```

Open http://127.0.0.1:8000 in your browser.

## Versioning

The docs use [mike](https://github.com/jimporter/mike) for multi-version deployment:

| Version | URL path  | Description                    |
|---------|-----------|--------------------------------|
| `dev`   | `/dev/`   | Development build (main branch)|
| `latest`| `/latest/`| Newest release (alias)         |
| `1.4`   | `/1.4/`   | Release v1.4.0                |

- **dev** – Always tracks `main`; deployed on every push to main.
- **latest** – Points to the most recent release; updated when a new release is published.
- **Versioned releases** – Each release (e.g., v1.4.0) is deployed to its `major.minor` path (e.g., `/1.4/`).

To preview all deployed versions locally (requires the `gh-pages` branch):

```bash
mike serve
```

To deploy a version manually:

```bash
mike deploy <version> [alias]...
```

For example: `mike deploy dev` or `mike deploy 1.4 latest --update-aliases`.

## Structure

The docs use:

- **MkDocs** – Static site generator
- **Material theme** – Navigation tabs, sections, integrated TOC, and version selector
- **mkdocstrings** – API reference generated from Python docstrings
- **mike** – Versioning and multi-version deployment

## CI Deployment

### Main branch (dev)

The [.github/workflows/docs.yml](https://github.com/dhruvdcoder/xlm-core/blob/main/.github/workflows/docs.yml) workflow:

1. Triggers on push to `main` or manual `workflow_dispatch`
2. Installs xlm-core, xlm-models, and idlm
3. Installs MkDocs, Material, mkdocstrings, and mike
4. Runs `mike deploy dev --push` to deploy to the `/dev/` path on the `gh-pages` branch

### Releases (versioned + latest)

The [.github/workflows/docs-release.yml](https://github.com/dhruvdcoder/xlm-core/blob/main/.github/workflows/docs-release.yml) workflow:

1. Triggers when a GitHub release is published
2. Checks out the release tag
3. Extracts the version (e.g., `v1.4.0` → `1.4`)
4. Runs `mike deploy 1.4 --update-aliases latest --push`
5. Runs `mike set-default latest --push` so the site root redirects to the latest release

**Prerequisite**: Enable GitHub Pages in repo Settings → Pages → Source: "Deploy from a branch" → Branch: `gh-pages` / `/(root)`.

## Dependencies

Install docs dependencies from `requirements/docs_requirements.txt`:

```
mkdocs>=1.6
mkdocs-material>=9.0
mkdocstrings[python]>=0.24
mike>=2.0
```
