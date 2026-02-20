# Building Docs

This page describes how the xLM documentation is built and deployed.

## Local Build


Install dependencies and build the docs:

```bash
pip install -e .
pip install -e xlm-models/
pip install -r requirements/docs_requirements.txt
mkdocs build
```

## Local Serve

To preview the docs with live reload (rebuilds on file save):

```bash
mkdocs serve
```

Open http://127.0.0.1:8000 in your browser. Use this for active editing—it watches for changes and rebuilds automatically.

**Live reload not working?** On network filesystems (e.g. NFS), file watchers often fail. Use this workaround—it restarts the server when files change:

```bash
pip install watchdog
watchmedo auto-restart -d . -p '*.md' -p '*.yml' --ignore-directories site -- mkdocs serve
```

Add `-a 0.0.0.0:8000` if you need to access from another machine. Refresh the browser after each change. On Linux, if you hit inotify limits, increase the watch limit: `echo 524288 | sudo tee /proc/sys/fs/inotify/max_user_watches`

## Versioning

The docs use [mike](https://github.com/jimporter/mike) for multi-version deployment:

| Version  | URL path   | Description                     |
|----------|------------|---------------------------------|
| `dev`    | `/dev/`    | Development build (main branch) |
| `latest` | `/latest/` | Newest release (alias)          |
| `1.4`    | `/1.4/`    | Release v1.4.0                  |

- **dev** – Always tracks `main`; deployed on every push to main.
- **latest** – Points to the most recent release; updated when a new release is published.
- **Versioned releases** – Each release (e.g., v1.4.0) is deployed to its `major.minor` path (e.g., `/1.4/`).

### Build and serve all versions locally

If `gh-pages` already has versions (e.g. from CI):

```bash
git fetch origin gh-pages
mike serve
```

Then open http://localhost:8000 and use the version selector. If you get a 404 at the root, run `mike set-default dev` (or `latest` if you have releases) before serving.

If `gh-pages` is empty or you want to test new versions locally before pushing:

```bash
# Checkout main for dev
git checkout main

# Install deps (if you already have, use the load the virtual environment)
pip install -e . -e xlm-models/ -r requirements/docs_requirements.txt

# Deploy dev locally (no --push)
mike deploy dev

# Set default so the root URL redirects (avoids 404 at /)
mike set-default dev

# Optional: deploy a release version (e.g. after checking out a tag)
git checkout v1.4.0  # or whatever tag
mike deploy 1.4 latest --update-aliases
mike set-default latest  # if you want root to redirect to latest
```

Then serve:

```bash
mike serve
```

**Note:** `mike serve` does not watch for file changes. After editing docs, run `mike deploy dev` (or the relevant version) again, then refresh the browser. For live reload during editing, use `mkdocs serve --livereload` instead—it only shows one version but rebuilds automatically on save.

If you see a 404 at the root URL, run `mike set-default <version>` (e.g. `mike set-default dev` or `mike set-default latest`) so the root redirects to that version. You can also navigate directly to a version path, e.g. `/dev/` or `/1.4/`.

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
watchdog>=3.0
```
