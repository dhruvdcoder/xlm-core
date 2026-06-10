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

## Related guides

Depending on what you are changing, these existing pages may help:

- [Data pipeline](../data-pipeline.md) — how `DatasetManager` and dataloader names connect to metrics
- [Metrics](../metrics.md) — defining and wiring metrics
- [Custom commands](../custom-commands.md) — extending `job_type` via external model `commands.yaml`
- [Adding a task or dataset](../adding-a-task.md) — if your core change includes a new benchmark
- [Debugging utilities](../debugging-utils.md) — debug modes and utilities

## Testing

- **Unit tests:** add or extend tests under {{ gh_dir('tests/core', 'tests/core/') }}.
- **Integration tests:** end-to-end datamodule / DDP scenarios live under {{ gh_dir('tests/integration', 'tests/integration/') }} — see [Integration tests](../../developers/testing/integration-tests.md).
- **Fast loop:** `pytest -m "not slow and not cli"` (see [Running tests](../../developers/testing/running-tests.md)).

## Work in progress

This guide will be expanded after maintainers walk through a concrete core contribution (Phase 4 of the contribution guidelines project). Until then, use the related guides above and existing code as reference.
