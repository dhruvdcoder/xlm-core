# Adding a maintained model

This guide covers adding a **first-party** model family under {{ gh_dir('xlm-models', 'xlm-models/') }} in this repository. For a model in a separate repo, see [External models](../external-models.md).

Reference implementations: `arlm`, `ilm`, `mlm`, `mdlm`, `flexmdm`, `dream`. Conceptual comparison: [Models overview](../../models/index.md).

## Quick start

```bash
xlm-scaffold my_family
```

This scaffolds Python modules, Hydra configs, and registers the family in {{ gh('xlm-models/xlm_models.json', 'xlm_models.json') }}. See [External models](../external-models.md#quick-start) for scaffold details (the same tool applies to in-repo families).

Install both packages in editable mode:

```bash
pip install -e .
pip install -e ./xlm-models
```

## Four components

Every working model implements four pieces that plug into the harness:

| Component | Role | Typical module |
|-----------|------|----------------|
| **Model** | Neural network and forward pass | `model_<family>.py` |
| **Loss** | Training objective | `loss_<family>.py` |
| **Predictor** | Inference / generation | `predictor_<family>.py` |
| **Collator** | Batch construction | `datamodule_<family>.py` |

You also typically add `types_<family>.py` (batch TypedDicts) and `metrics_<family>.py` (metric update functions).

Components for one family are designed to work together only with that family — not across families. See {{ gh('src/xlm/harness.py', 'harness.py') }} for `LossFunction` and `Predictor` protocols, and {{ gh('src/xlm/datamodule.py', 'datamodule.py') }} for `Collator`.

### Directory layout

```
xlm-models/<family>/
├── __init__.py
├── types_<family>.py
├── model_<family>.py
├── loss_<family>.py
├── predictor_<family>.py
├── datamodule_<family>.py
├── metrics_<family>.py
└── configs/
    ├── model/
    ├── model_type/
    ├── collator/
    ├── datamodule/
    └── experiment/
```

## Checklist

1. Implement the family under `xlm-models/<family>/` (mirror existing families).
2. Add Hydra configs under `xlm-models/<family>/configs/`.
3. Register in {{ gh('xlm-models/xlm_models.json', 'xlm_models.json') }} (`xlm-scaffold` does this).
4. Add tests under `tests/models/<family>/` using mixins in {{ gh('tests/models/_base.py', '_base.py') }} — see [Unit tests](../../developers/testing/unit-tests.md).
5. Optionally add `docs/models/<family>.md`, a `mkdocs.yml` nav entry, and an `api-autonav` module path.
6. Optionally add a CLI smoke entry in {{ gh('tests/cli/test_smoke.py', 'test_smoke.py') }} once a minimal experiment config exists.

## Hydra configuration

Configs for a model family live under `xlm-models/<family>/configs/`. Hydra discovers them via the external-models search path (see {{ gh('src/xlm/external_models.py', 'external_models.py') }}).

Logical composition:

```text
experiment
├── datamodule (+ collators per dataloader)
└── model components
    ├── model (architecture)
    ├── model_type (loss, predictor, metrics)
    └── noise_schedule, trainer, …
```

### Collator configs

A family often defines several collators:

1. **Base training** — unconditional LM batches (masking, padding, targets).
2. **Seq2seq training** — prefix + suffix in one batch.
3. **Seq2seq prediction** — prompt-only or prompt + separate target for metrics.

Example (ARLM default collator):

```yaml
# xlm-models/arlm/configs/collator/default_arlm.yaml
_target_: arlm.datamodule_arlm.DefaultARLMCollator
block_size: ${block_size}
tokenizer: ${global_components:tokenizer}
noise_schedule: ${global_components:noise_schedule}
```

### Datamodule config

Wire collators per split/dataloader name, e.g. `xlm-models/arlm/configs/datamodule/star_easy_arlm.yaml`:

```yaml
# @package _global_
defaults:
  - default
  - /collator@datamodule.dataset_managers.train.lm.collator: default_arlm
  - /collator@datamodule.dataset_managers.val.lm.collator: default_arlm

datamodule:
  print_batch_fn: arlm.datamodule_arlm.print_batch_arlm

tags:
  dataset: star_easy_arlm
```

### Model config

Architecture hyperparameters only, e.g. `xlm-models/<family>/configs/model/<family>.yaml`:

```yaml
# @package _global_
model:
  _target_: my_family.model_my_family.MyFamilyModel
  num_embeddings: ${tokenizer:full_vocab_size}
  d_model: 768
  # ...
tags:
  model: my_family_small
```

### Model type config

Loss, predictor, and default metrics — e.g. {{ gh('xlm-models/arlm/configs/model_type/arlm.yaml', 'arlm/configs/model_type/arlm.yaml') }}:

```yaml
# @package _global_
defaults:
  - /metrics@reported_metrics.train.lm.accumulated_loss: accumulated_loss
  - /metrics@reported_metrics.val.lm.accumulated_loss: accumulated_loss
  - /metrics@reported_metrics.test.lm.accumulated_loss: accumulated_loss

lightning_module:
  _target_: xlm.harness.Harness

loss:
  _target_: arlm.loss_arlm.ARLMLoss

predictor:
  _target_: arlm.predictor_arlm.ARLMPredictor
  tokenizer: ${lightning_module:tokenizer}
  noise_schedule: ${lightning_module:noise_schedule}
  max_steps: ${block_size}
  max_length: ${eval:${block_size}+${oc.select:input_block_size,0}}

reported_metrics:
  train:
    lm:
      accumulated_loss:
        prefix: train/lm
        update_fn: arlm.metrics_arlm.mean_metric_update_fn
  # val / test analogous
tags:
  model_type: arlm
```

### Experiment config

Compose overrides, e.g. {{ gh('xlm-models/arlm/configs/experiment/star_easy_arlm.yaml', 'arlm/configs/experiment/star_easy_arlm.yaml') }}:

```yaml
# @package _global_
defaults:
  - override /datamodule: star_easy_arlm
  - override /noise_schedule: dummy
  - override /model_type: arlm
  - override /model: rotary_transformer_small_arlm

per_device_batch_size: 64
block_size: 128
```

Run a smoke train:

```bash
xlm job_type=train job_name=my_family_debug experiment=star_easy_my_family debug=overfit
```

## Harness integration

{{ gh('src/xlm/harness.py', 'Harness') }} wires your components from config:

1. Instantiates the **model** from `model/` config.
2. Configures **loss** with model and tokenizer.
3. Configures **predictor** with model, tokenizer, and noise schedule.
4. Uses your **collator** via the datamodule.

## Testing

### Unit tests (required)

Follow the mixin pattern in [Unit tests](../../developers/testing/unit-tests.md):

- `tests/models/<family>/test_model_<family>.py` — inherit `BaseModelTests`
- `tests/models/<family>/test_loss_<family>.py` — inherit `BaseLossTests`
- `tests/models/<family>/test_collator_<family>.py` — inherit `BaseCollatorTests`
- Add predictor tests as needed

```bash
pytest tests/models/<family>/ -v
```

### Debug / smoke runs

Quick integration check without full training:

```bash
xlm job_type=train job_name=my_family_debug experiment=star_easy_my_family debug=overfit
```

For CI smoke coverage, append `(experiment, job_type)` to `SMOKE_RUNS` in {{ gh('tests/cli/test_smoke.py', 'test_smoke.py') }} — see [Running tests](../../developers/testing/running-tests.md#end-to-end-cli-smoke-tests).

## Example: ARLM

| Piece | Module | Config |
|-------|--------|--------|
| Model | {{ gh('xlm-models/arlm/model_arlm.py', 'model_arlm.py') }} | `configs/model/rotary_transformer_small_arlm.yaml` |
| Loss | {{ gh('xlm-models/arlm/loss_arlm.py', 'loss_arlm.py') }} | `configs/model_type/arlm.yaml` |
| Predictor | {{ gh('xlm-models/arlm/predictor_arlm.py', 'predictor_arlm.py') }} | `configs/model_type/arlm.yaml` |
| Collators | {{ gh('xlm-models/arlm/datamodule_arlm.py', 'datamodule_arlm.py') }} | `configs/collator/default_arlm.yaml`, `seq2seq_arlm.yaml`, `seq2seq_pred_arlm.yaml` |
| Experiment | — | {{ gh('xlm-models/arlm/configs/experiment/star_easy_arlm.yaml', 'star_easy_arlm.yaml') }} |

Narrative doc: [ARLM](../../models/arlm.md).

## Troubleshooting

| Problem | What to check |
|---------|----------------|
| `Unable to find or instantiate …` | Import the class manually: `python -c "from my_family.model_my_family import MyFamilyModel"` |
| Config not found | `configs/model/<family>.yaml` and `configs/model_type/<family>.yaml` exist; YAML `_target_` paths are correct |
| Model not discovered | Entry in `xlm_models.json`; `pip install -e ./xlm-models` |

See also [External models — Troubleshooting](../external-models.md#troubleshooting).

## Train an existing model on a new dataset

Use [Adding a task or dataset](../adding-a-task.md) to add preprocessing and dataset YAMLs under `src/xlm/configs/lightning_train/datasets/`, then add a datamodule config under `xlm-models/<family>/configs/datamodule/` that wires the new dataset and collators.
