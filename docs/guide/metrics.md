# Metrics

This guide covers how metrics are defined, configured, and evaluated.  For how
metrics connect to the data pipeline via **dataloader names**, see the
[Data Pipeline](data-pipeline.md) guide.

## Two Metric Categories

The `Harness` maintains two parallel metric dictionaries:

| Category | Purpose | Example |
|----------|---------|---------|
| `diagnostic_metrics` | Model-internal or debugging signals that may differ across model types. | per-token CE, length loss |
| `reported_metrics` | Standardised task-level metrics that are comparable across models. | accumulated loss, exact match, token accuracy |

Both use the same nested structure and the same `MetricWrapper` class; the only
difference is intent.  You can configure either (or both) as `null` if not
needed.

## MetricWrapper

Every configured metric is a `MetricWrapper` instance (`xlm.metrics`).  It
wraps a torchmetrics `Metric` and adds two pieces of glue:

- **`update_fn(batch, loss_dict, tokenizer, ...)`** -- a plain function
  (specified as a dotted-path string) that extracts the right inputs from the
  batch and loss dictionary and returns a `dict` of kwargs for the underlying
  `metric.update()`.
- **`log(pl_module, batch, loss_dict)`** -- logs the metric value via
  Lightning's `self.log()` with a configurable `prefix`, `on_step`, `on_epoch`,
  and `prog_bar`.

This separation means the `Metric` itself stays generic (e.g. `MeanMetric`,
`ExactMatch`) while all task-specific and model-specific extraction logic lives
in the `update_fn`.

## Storage Structure

Metrics are stored as nested `ModuleDict`s inside the `Harness`:

```
reported_metrics
  └── metrics_{stage}          (e.g. "metrics_val")
        └── {dataloader_name}  (e.g. "lm", "prediction")
              └── ModuleList[MetricWrapper, ...]
```

`diagnostic_metrics` follows the same layout.

## Step Flow

During every training / validation / test step, `Harness._step()`:

1. Resolves `dataloader_idx` to `dataloader_name` (see [Data Pipeline](data-pipeline.md)).
2. Calls `compute_loss(batch, ...)`.
3. Iterates over all `diagnostic_metrics` **and** `reported_metrics` for that
   dataloader name, calling `metric.update(...)` then `metric.log(...)` on each.

At epoch end, `on_validation_epoch_end` and `on_test_epoch_end` trigger
additional aggregate computations for dataloaders whose name contains
`"prediction"` -- specifically generative perplexity and any configured post-hoc
metrics.

## Configuration

Metric configs live in `configs/lightning_train/metrics/`.  Each file defines a
single `MetricWrapper`:

```yaml
# configs/lightning_train/metrics/accumulated_loss.yaml
_target_: xlm.metrics.MetricWrapper
name: accumulated_loss
metric:
  _target_: torchmetrics.MeanMetric
prefix: ???   # e.g. train/lm, val/lm
update_fn: ??? # e.g. mdlm.metrics_mdlm.mean_metric_update_fn
```

These are composed into the model-type config via Hydra defaults.  For example,
a seq2seq model type might wire up both diagnostic and reported metrics like
this:

```yaml
# model_type config (simplified)
defaults:
  - /metrics@reported_metrics.train.lm.accumulated_loss: accumulated_loss
  - /metrics@reported_metrics.val.lm.accumulated_loss: accumulated_loss
  - /metrics@reported_metrics.val.prediction.exact_match: seq2seq_exact_match
  - /metrics@reported_metrics.val.prediction.token_accuracy: seq2seq_token_accuracy
  - /metrics@diagnostic_metrics.train.lm.length_loss: seq2seq_length_loss
  - /metrics@diagnostic_metrics.train.lm.token_ce: seq2seq_token_ce

diagnostic_metrics:
  train:
    lm:
      length_loss:
        prefix: train/lm
        update_fn: idlm.metrics.length_loss_metric_update_fn
      token_ce:
        prefix: train/lm
        update_fn: idlm.metrics.token_ce_metric_update_fn

reported_metrics:
  train:
    lm:
      accumulated_loss:
        prefix: train/lm
        update_fn: idlm.metrics.mean_metric_update_fn
  val:
    lm:
      accumulated_loss:
        prefix: val/lm
        update_fn: idlm.metrics.mean_metric_update_fn
    prediction:
      exact_match:
        prefix: val/prediction
        update_fn: idlm.metrics.seq2seq_exact_match_update_fn
      token_accuracy:
        prefix: val/prediction
        update_fn: idlm.metrics.seq2seq_token_accuracy_update_fn
```

The top-level keys (`train`, `val`, `test`) and the second-level keys (`lm`,
`prediction`) must match the dataloader names in the datamodule config -- that
is how the `Harness` knows which metrics to apply to which dataloader.

## Built-in Metrics and Update Functions

| Metric class | Module | Description |
|-------------|--------|-------------|
| `MeanMetric` | `torchmetrics` | Tracks a running mean (used for loss). |
| `ExactMatch` | `xlm.metrics` | Sequence-level exact match rate. |
| `TokenAccuracy` | `xlm.metrics` | Per-token accuracy over predicted positions. |

| Update function | Module | Extracts |
|----------------|--------|----------|
| `mean_metric_update_fn` | `xlm.metrics` | `loss_dict["batch_loss"]` |
| `exact_match_update_fn` | `xlm.metrics` | `batch["target_ids"]` vs `loss_dict["ids"]` |
| `seq2seq_exact_match_update_fn` | `xlm.metrics` | Concatenated input+target vs predicted ids |
| `seq2seq_token_accuracy_update_fn` | `xlm.metrics` | Token-level accuracy on target positions |

To add a custom metric, create a new `Metric` subclass (or use an existing
torchmetrics metric), write an `update_fn` that extracts the right fields, add a
YAML config under `configs/lightning_train/metrics/`, and wire it into your
model-type config.
