# Adding a new task or dataset

This guide walks through wiring a Hugging Face dataset (or similar source) into xlm-core: Python preprocessing in `xlm.tasks`, Hydra configs under `configs/lightning_train/`, and alignment between **dataloader names** and **metrics**.

Read these guides first if you have not worked with the pipeline before:

- [Data Pipeline](data-pipeline.md) — how `TextDataModule`, `DatasetManager`, and dataloader names connect to the `Harness`.
- [Metrics](metrics.md) — `MetricWrapper`, `reported_metrics`, and prediction/post-hoc behavior.
- [Evaluate a model](eval.md) — loading checkpoints for evaluation jobs.

Cursor users may also invoke the **`xlm-create-task`** Agent Skill (`~/.cursor/skills/xlm-create-task/`) for a condensed checklist and copied excerpts.

---

## 1. Pick a wiring pattern

| Pattern | Use when | Typical config |
|---------|----------|----------------|
| **Standard LM / map dataset** | Hub dataset; preprocess once and cache via `DatasetManager` | `datasets/default_text.yaml` defaults → [`DatasetManager`](../../src/xlm/datamodule.py) |
| **Streaming / sequence packing** | Huge corpus; packed blocks without per-example padding | `iterable_dataset_shards`, `on_the_fly_group_processor: xlm.datamodule.pack_sequences` (see [Data Pipeline](data-pipeline.md)) |
| **Eval-only / benchmarks** | Smaller splits; avoid manual `save_to_disk` cache semantics | `datasets/default_eval.yaml` → [`EvalDatasetManager`](../../src/xlm/datamodule.py) |
| **Heavy optional dependencies** | Chemistry stacks, etc. | Lazy facade package that imports implementation on demand ([`safe_molgen/__init__.py`](../../src/xlm/tasks/safe_molgen/__init__.py)); document `pip install "xlm-core[...]"` |

---

## 2. Implement task code (`src/xlm/tasks/<your_task>/__init__.py`)

Put each task in its **own directory** under [`tasks/`](../../src/xlm/tasks/) with an `__init__.py` that defines preprocess functions, filters, evaluators, etc.

Hydra resolves callables by **dotted path** (for example `xlm.tasks.owt.preprocess_fn`). You do **not** need to list every symbol in the parent [`tasks/__init__.py`](../../src/xlm/tasks/__init__.py)—that file is only a **package marker**, not a barrel re-export of all tasks.

### Preprocess function

Configured as `preprocess_function` on the dataset manager. At runtime it is loaded with `get_function` and passed to `datasets.Dataset.map`. The effective signature is:

```text
(example: dict, tokenizer, **preprocess_function_kwargs) -> dict
```

Return only the columns your downstream collator and model need (drop raw text via `columns_to_remove`, or restrict with `columns_to_keep` on eval configs).

**Minimal LM-style preprocessing** — encode raw text to token IDs; combine with `on_the_fly_processor: xlm.datamodule.token_ids_to_input_ids`:

- Code: [`owt/__init__.py`](../../src/xlm/tasks/owt/__init__.py)
- Dataset YAML: [`datasets/owt_train.yaml`](../../src/xlm/configs/lightning_train/datasets/owt_train.yaml)

**Structured fields + different processors per split** — build task-specific columns (for example `input_token_ids`, `prompt_token_ids`), then choose `on_the_fly_processor` per dataset YAML:

- Code: [`sudoku_extreme/__init__.py`](../../src/xlm/tasks/sudoku_extreme/__init__.py)
- Train: [`datasets/sudoku_extreme_train.yaml`](../../src/xlm/configs/lightning_train/datasets/sudoku_extreme_train.yaml)
- Prediction split overrides processor via defaults chain: [`datasets/sudoku_extreme_val_pred.yaml`](../../src/xlm/configs/lightning_train/datasets/sudoku_extreme_val_pred.yaml)

**Optional row filters** — set `filter_fn` to a dotted path and **`filter_suffix`** (required when `filter_fn` is set). Example filter implementations live beside preprocessing in [`sudoku_extreme/__init__.py`](../../src/xlm/tasks/sudoku_extreme/__init__.py).

**Benchmark-style eval + post-hoc scoring** — preprocessing builds prompt/target columns and carries gold answers through the batch for logging; a **`post_hoc_evaluator`** class scores saved predictions:

- Code: [`math500/__init__.py`](../../src/xlm/tasks/math500/__init__.py) (`math500_preprocess_fn`, `Math500Eval`)
- Eval dataset: [`datasets/math500_test.yaml`](../../src/xlm/configs/lightning_train/datasets/math500_test.yaml)

**Large optional imports** — keep the default import surface light and defer heavy imports (see [`safe_molgen/__init__.py`](../../src/xlm/tasks/safe_molgen/__init__.py) delegating to [`_safe_molgen_impl.py`](../../src/xlm/tasks/safe_molgen/_safe_molgen_impl.py)).

For larger tasks you may instead split logic into additional modules inside `tasks/<your_task>/` and re-export public symbols from `__init__.py`; Hydra paths stay `xlm.tasks.<your_task>.<symbol>`.

---

## 3. Add dataset YAML (`src/xlm/configs/lightning_train/datasets/`)

Create a new file or compose existing ones with Hydra `defaults:`.

1. **Training / general `DatasetManager`** — start from [`default_text.yaml`](../../src/xlm/configs/lightning_train/datasets/default_text.yaml):

   - `full_name`: Hub path in the form `namespace/dataset_name/split` (last segment is the split passed to `load_dataset`).
   - `full_name_debug`: optional smaller split used when `DEBUG_OVERFIT` is enabled.
   - `preprocess_function`, `preprocess_function_kwargs`.
   - `on_the_fly_processor` / `on_the_fly_group_processor` (mutually exclusive with group processors per [`DatasetManager`](../../src/xlm/datamodule.py)).
   - `columns_to_remove` or `columns_to_keep`.
   - `stages`: which Lightning stages use this manager (`fit`, `validate`, `test`, `predict`).
   - `collator: ???` — resolved by experiment/model package via `/collator@...` overrides.

2. **Eval-only `EvalDatasetManager`** — start from [`default_eval.yaml`](../../src/xlm/configs/lightning_train/datasets/default_eval.yaml):

   - No manual disk cache path; HF map cache optional via `load_from_cache_file`.
   - Typical `stages: [validate, test]`.

Reuse patterns from [`math500_test.yaml`](../../src/xlm/configs/lightning_train/datasets/math500_test.yaml) for `columns_to_keep` and preprocess kwargs.

---

## 4. Compose a datamodule config

Datamodule skeleton: [`datamodule/default.yaml`](../../src/xlm/configs/lightning_train/datamodule/default.yaml).

Wire each `(split, dataloader_name)` to your dataset group using Hydra defaults, for example:

```yaml
defaults:
  - default
  - /datasets@datamodule.dataset_managers.train.lm: owt_train
  - /datasets@datamodule.dataset_managers.val.lm: owt_val
```

Full example: [`datamodule/owt.yaml`](../../src/xlm/configs/lightning_train/datamodule/owt.yaml).

If you introduce a **new** logical evaluation stream (for example `math500_prediction`), pick a **new dataloader name** and use it consistently across val/test **and** metrics:

- Example: [`xlm-models/dream/configs/datamodule/math500_dream.yaml`](../../xlm-models/dream/configs/datamodule/math500_dream.yaml)

Set `tags.dataset` in the experiment/datamodule root when you want W&B or logging tags (same file).

---

## 5. Wire collators

Dataset YAMLs leave `collator: ???`. Resolve it from the model/experiment package:

```yaml
- /collator@datamodule.dataset_managers.val.math500_prediction.collator: math500_pred_dream
```

The collator must produce batch tensors your **model** and **metric update functions** expect.

---

## 6. Wire metrics and optional post-hoc evaluation

Under `reported_metrics.<train|val|test>.<dataloader_name>`, each nested dict entry is a configured metric (typically `MetricWrapper`). **The `<dataloader_name>` keys must match** the keys under `datamodule.dataset_managers` for that stage — see [Metrics](metrics.md).

Hydra composition often looks like:

```yaml
defaults:
  - /metrics@reported_metrics.val.lm.accumulated_loss: accumulated_loss
```

For dataloaders whose name contains **`prediction`**, epoch-end hooks run extra aggregates (generative perplexity and configured post-hoc evaluators); see [Metrics](metrics.md).

Post-hoc evaluator `_target_` example:

```yaml
post_hoc_evaluator:
  _target_: xlm.tasks.math500.Math500Eval
```

Experiment reference: [`xlm-models/dream/configs/experiment/math500_dream_eval.yaml`](../../xlm-models/dream/configs/experiment/math500_dream_eval.yaml).

---

## 7. Dependencies and extras

If your task needs packages beyond core xlm-core, document them in the task module docstring and add or reference an optional extra in project requirements (for example `"xlm-core[safe]"`, files under [`requirements/`](../../requirements/)).

---

## 8. Verify

1. Run `job_type=prepare_data` (rank 0 downloads and preprocesses/caches where applicable).
2. Smoke train or eval with the smallest experiment config you can.
3. Use `DEBUG_OVERFIT` and `full_name_debug` to force train-shaped data into val/test during debugging ([`DatasetManager`](../../src/xlm/datamodule.py)).
4. Add or extend integration coverage under `tests/integration/datamodule/` when behavior is easy to regress.

---

## Quick reference table

| Artifact | Location |
|----------|----------|
| Task preprocessing / filters / evaluators | `src/xlm/tasks/<task>/` (one directory per task) |
| Dataset group YAML | `src/xlm/configs/lightning_train/datasets/` |
| Datamodule composition | `src/xlm/configs/lightning_train/datamodule/` (+ model repos under `xlm-models/`) |
| Metric snippets | `src/xlm/configs/lightning_train/metrics/` |
| Pipeline implementation | `src/xlm/datamodule.py` |
