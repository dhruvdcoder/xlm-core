# Running your model on your task

Use XLM as infrastructure to train your model on your dataset — without modifying xlm-core. Hydra resolves any importable Python callable by dotted path, so your preprocess function, collators, and metrics can live in your own package.

**Built-in reference:** STAR-easy + ARLM — {{ gh('xlm-models/arlm/configs/datamodule/star_easy_arlm.yaml', 'datamodule') }}, {{ gh('xlm-models/arlm/configs/experiment/star_easy_arlm.yaml', 'experiment') }}, {{ gh('xlm-models/arlm/configs/datamodule/star_arlm.yaml', 'skeleton') }}.

---

## What you need

1. **Task code** — a preprocess function (and optional evaluators) in any importable module, e.g. `my_package.my_task.preprocess_fn`
2. **Dataset YAMLs** — under your model's `configs/datasets/` (or `src/xlm/configs/…` if contributing upstream)
3. **Datamodule + experiment configs** — under your model's `configs/`

Point Hydra at your callables in YAML:

```yaml
preprocess_function: my_package.my_task.preprocess_fn
```

No changes to xlm-core source required.

---

## 1. Datamodule config

`xlm-models/<family>/configs/datamodule/<task>_<family>.yaml`

Extend the family skeleton and swap dataset pointers:

```yaml
# @package _global_
defaults:
  - star_arlm                                    # inherits collators + print_batch_fn
  - /datasets@datamodule.dataset_managers.train.lm: star_easy_train
  - /datasets@datamodule.dataset_managers.val.lm: star_easy_val
  - /datasets@datamodule.dataset_managers.val.prediction: star_easy_val_pred
  - /datasets@datamodule.dataset_managers.test.lm: star_easy_test
  - /datasets@datamodule.dataset_managers.test.prediction: star_easy_test_pred
  - /datasets@datamodule.dataset_managers.predict.prediction: star_easy_test_pred

tags:
  dataset: star_easy
```

- The skeleton provides collators — don't duplicate them.
- `<stage>.<dataloader_name>` must match `stages` in your dataset YAMLs.
- New dataloader names need matching entries in `reported_metrics` ([Metrics](../metrics.md)).

## 2. Experiment config

`xlm-models/<family>/configs/experiment/<task>_<family>.yaml`

```yaml
# @package _global_
defaults:
  - override /datamodule: star_easy_arlm
  - override /noise_schedule: dummy
  - override /model_type: arlm_seq2seq
  - override /model: rotary_transformer_small_arlm

per_device_batch_size: 64
block_size: 14
monitored_metric: val/lm/accumulated_loss
```

Copy hyperparameters from the closest existing experiment; tokenizer and block sizes must match your task.

## 3. Verify

```bash
xlm job_type=prepare_data job_name=prepare_star_easy_arlm experiment=star_easy_arlm debug=overfit
xlm job_type=train job_name=star_easy_arlm_debug experiment=star_easy_arlm debug=overfit
```

Both should exit 0. Stop the train once loss is decreasing and val batches load.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `preprocess_function` not found | Ensure your package is installed (`pip install -e .`) and the dotted path is correct |
| Config not found | Dataset YAMLs must be on Hydra's config search path; reinstall your model package |
| Experiment not found | Config in `xlm-models/<family>/configs/experiment/`; run `pip install -e ./xlm-models` |
| Collator fails | Family skeleton must be first in `defaults` |
| Metrics mismatch | `reported_metrics` keys must mirror `datamodule.dataset_managers` keys |

## See also

[Adding a task (maintained)](../adding-a-task.md) · [External models](../external-models.md) · [Data pipeline](../data-pipeline.md)
