# Adding a new task or dataset

Adding a task has two phases:

| Phase | What | Where (maintained tasks) |
|-------|------|--------------------------|
| **A** | Preprocess function + dataset YAMLs | `src/xlm/tasks/<task>/`, `src/xlm/configs/lightning_train/datasets/` |
| **B** | Datamodule + experiment configs | `xlm-models/<family>/configs/` (see [Running your model on your task](../contributing/adding-a-task-external.md)) |

For tasks shipped with xlm-core, preprocessing lives in `src/xlm/tasks/`. For external tasks, Hydra resolves any importable dotted path (e.g. `my_package.my_task.preprocess_fn`) — your code does not need to live inside this repo.

**Reference example:** STAR-easy + ARLM — {{ gh('src/xlm/tasks/star/__init__.py', 'task code') }}, {{ gh('src/xlm/configs/lightning_train/datasets/star.yaml', 'base YAML') }}, split YAMLs `star_easy_{train,val,val_pred,test,test_pred}.yaml`, {{ gh('xlm-models/arlm/configs/datamodule/star_easy_arlm.yaml', 'datamodule') }}, {{ gh('xlm-models/arlm/configs/experiment/star_easy_arlm.yaml', 'experiment') }}.

**Prerequisites:** [Data Pipeline](data-pipeline.md), [Metrics](metrics.md), [Evaluate](eval.md).

---

## 1. Pick a wiring pattern

| Pattern | When | Start from |
|---------|------|------------|
| Standard LM / map dataset | Hub dataset, preprocess + cache | {{ gh('src/xlm/configs/lightning_train/datasets/default_text.yaml', 'default_text.yaml') }} |
| Streaming / sequence packing | Huge corpus, packed blocks | `on_the_fly_group_processor: xlm.datamodule.pack_sequences` |
| Eval-only / benchmarks | Small splits, no disk cache | {{ gh('src/xlm/configs/lightning_train/datasets/default_eval.yaml', 'default_eval.yaml') }} |
| Heavy optional deps | Chemistry stacks, etc. | Lazy facade ({{ gh('src/xlm/tasks/safe_molgen/__init__.py', 'safe_molgen') }}), `pip install "xlm-core[...]"` |

---

## 2. Task code

**Maintained tasks:** one directory per task under {{ gh_dir('src/xlm/tasks', 'tasks/') }} with an `__init__.py`.

**External tasks:** same pattern in your own package — set `preprocess_function: my_package.my_task.preprocess_fn` in your dataset YAML.

Hydra resolves callables by dotted path; no need to re-export from a parent `__init__.py`.

### Preprocess function

Signature: `(example: dict, tokenizer, **kwargs) -> dict`. Set as `preprocess_function` on the dataset manager; passed to `datasets.Dataset.map` at runtime.

Drop unneeded columns via `columns_to_remove` or `columns_to_keep`.

**By example:**

| Pattern | Code | Config |
|---------|------|--------|
| Minimal LM (text → token IDs) | {{ gh('src/xlm/tasks/owt/__init__.py', 'owt') }} | {{ gh('src/xlm/configs/lightning_train/datasets/owt_train.yaml', 'owt_train.yaml') }} |
| Structured seq2seq (STAR) | {{ gh('src/xlm/tasks/star/__init__.py', 'star') }} | {{ gh('src/xlm/configs/lightning_train/datasets/star_easy_train.yaml', 'star_easy_train.yaml') }} |
| Row filters | {{ gh('src/xlm/tasks/sudoku_extreme/__init__.py', 'sudoku_extreme') }} | set `filter_fn` + `filter_suffix` |
| Post-hoc eval (Math500) | {{ gh('src/xlm/tasks/math500/__init__.py', 'math500') }} | {{ gh('src/xlm/configs/lightning_train/datasets/math500_test.yaml', 'math500_test.yaml') }} |
| Code-execution eval (GSM8K) | {{ gh('src/xlm/tasks/tinygsm/gsm8k.py', 'gsm8k') }} | [TinyGSM runbook](../tasks/tinygsm_gsm8k.md) |
| Heavy optional imports | {{ gh('src/xlm/tasks/safe_molgen/__init__.py', 'safe_molgen') }} → {{ gh('src/xlm/tasks/safe_molgen/_safe_molgen_impl.py', '_impl') }} | defer imports |

---

## 3. Dataset YAMLs (`src/xlm/configs/lightning_train/datasets/`)

Compose from existing base configs with Hydra `defaults:`.

**For training** ({{ gh('src/xlm/configs/lightning_train/datasets/default_text.yaml', 'default_text.yaml') }}):

- `full_name` — Hub path: `namespace/dataset_name/split`
- `full_name_debug` — smaller split for `DEBUG_OVERFIT`
- `preprocess_function`, `preprocess_function_kwargs`
- `on_the_fly_processor` / `on_the_fly_group_processor` (mutually exclusive)
- `columns_to_remove` or `columns_to_keep`
- `stages` — which Lightning stages use this manager (`fit`, `validate`, `test`, `predict`)
- `collator: ???` — resolved by the model datamodule

**For eval-only** ({{ gh('src/xlm/configs/lightning_train/datasets/default_eval.yaml', 'default_eval.yaml') }}): no disk cache, typical `stages: [validate, test]`.

---

## 4. Wire to a model

See [Running your model on your task](../contributing/adding-a-task-external.md). In short: extend a family skeleton, swap `/datasets@…` pointers, add an experiment config.

```yaml
# xlm-models/arlm/configs/datamodule/star_easy_arlm.yaml
defaults:
  - star_arlm                                        # inherits collators
  - /datasets@datamodule.dataset_managers.train.lm: star_easy_train
  - /datasets@datamodule.dataset_managers.val.lm: star_easy_val
  # … remaining splits
tags:
  dataset: star_easy
```

---

## 5. Metrics and post-hoc evaluation

`reported_metrics.<stage>.<dataloader_name>` keys **must match** `datamodule.dataset_managers` keys. See [Metrics](metrics.md).

Dataloaders named `*prediction*` trigger generative-perplexity and post-hoc evaluator hooks at epoch end.

```yaml
post_hoc_evaluator:
  _target_: xlm.tasks.math500.Math500Eval
```

See [Post-hoc evaluation](eval.md#post-hoc-evaluation) for details.

---

## 6. Dependencies

If your task needs extra packages, document them in the module docstring and add an optional extra (e.g. `"xlm-core[safe]"`) under {{ gh_dir('requirements', 'requirements/') }}.

---

## 7. Verify

```bash
xlm job_type=prepare_data job_name=prepare_star_easy_arlm experiment=star_easy_arlm debug=overfit
xlm job_type=train job_name=star_easy_arlm_debug experiment=star_easy_arlm debug=overfit
```

Both should exit 0. Stop the train once you see loss decreasing and val batches loading.

| Problem | Fix |
|---------|-----|
| `cannot import name 'no_init_weights'` | Pin `transformers<5` |
| `collator: ???` unresolved | Wire collators in model datamodule, not dataset YAMLs |
| Duplicate model warnings | Harmless when `xlm-models/` is both on disk and editable-installed |

---

## Quick reference

| Artifact | Location |
|----------|----------|
| Task code | `src/xlm/tasks/<task>/` (maintained) or your own package |
| Dataset YAMLs | `src/xlm/configs/lightning_train/datasets/` |
| Datamodule composition | `src/xlm/configs/lightning_train/datamodule/` + `xlm-models/` |
| Metric snippets | `src/xlm/configs/lightning_train/metrics/` |
| Pipeline implementation | `src/xlm/datamodule.py` |

**More examples:** [ARLM](../../models/arlm.md), [TinyGSM](../tasks/tinygsm.md) ([FlexMDM](../models/flexmdm.md#tinygsm), [MLM](../models/mlm.md#tinygsm), [ARLM](../models/arlm.md#tinygsm)).
