# Docs, bugs, and issues

You can contribute without touching model or task code: improve documentation, file clear bug reports, or propose features.

## Improving documentation

### Build the site locally

**Prerequisite:** complete [Install from source](https://github.com/dhruvdcoder/xlm-core/blob/main/CONTRIBUTING.md#install-from-source) in the Contributing hub (includes docs dependencies).

From the repository root:

```bash
mkdocs build
```

Preview with live reload:

```bash
mkdocs serve
```

### Add or update a page

1. Edit or create Markdown under {{ gh_dir('docs', 'docs/') }}.
2. Add an entry to the `nav:` section in {{ gh('mkdocs.yml', 'mkdocs.yml') }} so the page appears in the sidebar.
3. Run `mkdocs build` to confirm the site builds.

### Source links in docs

The docs use MkDocs macros defined in {{ gh('main.py', 'main.py') }} at the repo root:

- `{{ gh('path/to/file.py') }}` — link to a file on GitHub
- `{{ gh_dir('path/to/dir') }}` — link to a directory on GitHub

Example in a doc source file:

```markdown
See {{ gh('src/xlm/harness.py', 'harness.py') }} for the Lightning module.
```

## Reporting bugs

Before opening a bug report:

1. **Search [Issues](https://github.com/dhruvdcoder/xlm-core/issues)** for duplicates.
2. If you are unsure whether it is a library bug or your environment, ask in [Q&A Discussions](https://github.com/dhruvdcoder/xlm-core/discussions/categories/q-a).

When filing a bug, use the [bug report template](https://github.com/dhruvdcoder/xlm-core/issues/new?template=bug_report.md) and include:

- A **minimal repro script** or exact `xlm` command
- **Versions** (`xlm-core`, PyTorch, Lightning, Hydra) — e.g. `python -c "import xlm; print(xlm.__version__)"`
- **Relevant Hydra config** (`experiment=...` and any overrides)
- Expected vs actual behavior

## Feature requests

Use the [feature request template](https://github.com/dhruvdcoder/xlm-core/issues/new?template=feature_request.md). Describe the problem, your proposed solution, and which component is affected (core, model, task, docs, CLI).

For new model families or datasets, prefer the dedicated templates:

- [New model proposal](https://github.com/dhruvdcoder/xlm-core/issues/new?template=new_model.md)
- [New task / dataset proposal](https://github.com/dhruvdcoder/xlm-core/issues/new?template=new_task.md)

## Questions

Use [Discussions](https://github.com/dhruvdcoder/xlm-core/discussions) or the [question issue template](https://github.com/dhruvdcoder/xlm-core/issues/new?template=question.md) and note what you have already tried.
