# Evaluate a model

Evaluation uses `job_type=eval` with the same `experiment=...` (and related Hydra defaults) as training so the `Harness`, datamodule, and tokenizer match the checkpoint or Hub layout.

Weights are resolved by `load_model_for_inference()` with **`config_prefix=eval`** (see `src/xlm/utils/model_loading.py`). You can load from, in priority order:

1. **Full Lightning checkpoint** (`.ckpt`) — weights plus training metadata. Set **`eval.ckpt_path`** or **`eval.checkpoint_path`** (both are supported; `ckpt_path` is checked first).
2. **Model-only checkpoint** (`.pt` / state dict) — weights only. Extract from a full checkpoint with `xlm job_type=extract_checkpoint`, then set **`eval.model_only_checkpoint_path`**.
3. **Hugging Face Hub** — set **`hub.repo_id`** (e.g. `username/model-name`). Optionally pin **`hub.revision`** (branch, tag, or commit; default is `main`).

If a **full** checkpoint path is set and valid, it always wins; **`hub.repo_id`** and **`eval.model_only_checkpoint_path`** are not used in that case.

!!! note "`hub_checkpoint_path` is not for `eval`"
    The top-level **`hub_checkpoint_path`** key is used by **`job_type=push_to_hub`**, not by `eval`. For evaluation from the Hub, use **`+hub.repo_id=...`** only (plus optional **`+hub.revision=...`**).

## Authentication (Hub only)

Set a token so weights can be downloaded:

- Export **`HF_HUB_KEY`**, or add it to **`.secrets.env`** (loaded automatically by the `xlm` entrypoint).

## Examples

**Local full checkpoint:**

```bash
xlm job_type=eval job_name=lm1b_eval experiment=lm1b_ilm \
  +eval.ckpt_path=/path/to/epoch=0-step=1000.ckpt
```

**Local model-only weights:**

```bash
xlm job_type=eval job_name=lm1b_eval experiment=lm1b_ilm \
  +eval.model_only_checkpoint_path=/path/to/model.pt
```

**Weights from the Hub (no local `.ckpt` path):**

```bash
xlm job_type=eval job_name=lm1b_eval experiment=lm1b_ilm \
  +hub.repo_id=username/my-model \
  +hub.revision=main
```

## Fallback checkpoints vs Hub

If you do **not** set `eval.ckpt_path`, `eval.checkpoint_path`, or `eval.model_only_checkpoint_path`, eval still looks under **`checkpointing_dir`** for **`best.ckpt`** or **`last.ckpt`**. If either exists, that **full** checkpoint is loaded and **Hub weights are ignored**.

For a pure Hub eval, run with an output layout where those files are absent, or override **`checkpointing_dir`** to a directory without checkpoints.

## Post-hoc evaluation

Some task metrics need the **full set of generations** for a split (for example sequence-level scores, diversity, or MAUVE). Those run **after** the epoch finishes, in addition to per-step `MetricWrapper` metrics. Step-level prediction behavior is summarized in [Metrics](metrics.md) (“prediction” dataloaders); this section describes the **saved JSONL → post-hoc `eval()`** path.

### When it runs

On **rank 0**, at the end of **validation** or **test**, the `Harness` loops over each dataloader whose name contains the substring **`prediction`**. For each such name it calls [`compute_post_hoc_metrics`](../../src/xlm/harness.py) if `cfg.post_hoc_evaluator` is set. The **`predict`** stage does the same for the `unconditional_prediction` dataloader (see `on_predict_end` in the same file).

If `cfg.post_hoc_evaluator` is **`null`**, the call returns immediately (no file read, no `eval()`).

**Generative perplexity** (judge causal LMs over logged `text`) is implemented as post-hoc only: use [`GenerativePerplexityPostHocEval`](../../src/xlm/tasks/owt/generative_perplexity_post_hoc.py) via packaged YAML under **`post_hoc_evaluator/`** (e.g. `gen_ppl_gpt2_large`). The legacy top-level `generative_perplexity` config key was **removed**; using it raises a clear `ValueError` at init.

**`force_predict`** (on `lightning_module`, default `true`): when set to `false`, if the prediction JSONL for the current `(split, dataloader_name, epoch, step)` already exists and is non-empty, the harness skips **writing** new rows for that step (no extra forward logging) but still runs post-hoc by reading that file. Use paths consistent with `FilePredictionWriter` naming.

### Weights & Biases prediction tables

Lightning `log_text` prediction tables receive rows at **batch time** (from `predictor.to_dict`), **before** post-hoc runs. Per-example fields added by `post_hoc_evaluator.eval()` appear in **`results_{epoch}_{step}.json`**, not in the default W&B table.

### Predictions on disk (`LogPredictions`)

During the run, [`LogPredictions`](../../src/xlm/log_predictions.py) (via a **`FilePredictionWriter`**) appends one JSON object per line under:

`{run_dir}/predictions/{split}/{dataloader_name}/{epoch=…}_{step=…}.jsonl`

Each line is built from `predictor.to_dict(...)`, optional **`additional_fields_from_batch`**, and—when the writer is invoked with ground-truth text—a per-row **`truth`** field. **`compute_post_hoc_metrics`** reloads that file with `LogPredictions.read(...)` and passes the resulting `List[dict]` to the evaluator.

Your evaluator should read the keys your **collator + predictor** actually write (`text`, `truth`, `target_ids`, task-specific fields, etc.). Post-hoc code may **add** keys to each row; those enriched rows are written only to the **results JSON** (below)—the source **`.jsonl` is not modified** by `compute_post_hoc_metrics`.

### Single evaluator or composite chain

Hydra instantiates **one** object from the top-level `post_hoc_evaluator` config ([`setup_post_hoc_evaluator`](../../src/xlm/harness.py)). Use [`CompositePostHocEvaluator`](../../src/xlm/tasks/composite_eval/__init__.py) when you need **several** post-hoc passes on the same dataloader (e.g. MAUVE then generative perplexity). For each matching dataloader-name pattern, the value may be a **single** evaluator, an **ordered dict** of named sub-evaluators (run order = YAML / dict key order), or an **ordered list**; chain entries run sequentially, threading `predictions` through each `eval()` and merging `aggregated_metrics` (duplicate metric keys: **last wins**, with a warning).

### `eval()` contract

Your class should implement:

`eval(predictions, tokenizer=None, **kwargs) -> (predictions, aggregated_metrics)`

The harness passes `tokenizer=self.tokenizer` and `dataloader_name=...` (evaluators may ignore extra kwargs). **`aggregated_metrics`** is logged with `self.log(f"{split}/{metric_name}", value, ...)`, so values must be **numeric scalars** (floats/ints). Strings or nested structures will not work as Lightning-logged metrics.

Full results (aggregated dict **plus** per-row dicts, including any fields your `eval` added) are written beside the predictions tree:

`{run_dir}/predictions/{split}/{dataloader_name}/results_{epoch=…}_{step=…}.json`

The original **`.jsonl` is not modified** by post-hoc (unlike older generative-perplexity behavior that could append enriched rows).

### Hydra configuration

Packaged snippets live under [`src/xlm/configs/lightning_train/post_hoc_evaluator/`](../../src/xlm/configs/lightning_train/post_hoc_evaluator/). Base training config loads **`post_hoc_evaluator/default.yaml`**: an empty [`CompositePostHocEvaluator`](../../src/xlm/tasks/composite_eval/__init__.py). Each other file in that folder defines **one** evaluator object; compose it under a dataloader-name **pattern** (substring match) and a short **alias** using a structured defaults entry (or equivalent YAML / CLI overrides).

| Config (file)                                                     | Role                                                             |
|-------------------------------------------------------------------|------------------------------------------------------------------|
| `default`                                                         | Empty composite (`evaluators: {}`)                               |
| `denovo`                                                          | Small-molecule de novo metrics (`DeNovoEval`)                    |
| `syntactic`                                                       | Syntactic validity (`SyntacticMetricsEvaluator`)                 |
| `mauve_text`                                                      | Text MAUVE (`MauveTextEval`)                                     |
| `gen_ppl_gpt2_large` / `gen_ppl_gpt2_small` / `gen_ppl_llama3_3b` | Generative perplexity judges (`GenerativePerplexityPostHocEval`) |

**Compose from `defaults`** (pattern `prediction` matches typical `*_prediction` dataloaders):

```yaml
defaults:
  - /post_hoc_evaluator@post_hoc_evaluator.evaluators.prediction.mauve: mauve_text
```

Add a second evaluator on the same loader:

```yaml
defaults:
  - /post_hoc_evaluator@post_hoc_evaluator.evaluators.prediction.mauve: mauve_text
  - /post_hoc_evaluator@post_hoc_evaluator.evaluators.prediction.gen_ppl: gen_ppl_gpt2_large
```

**CLI** (same idea; quote if your shell splits on dots):

```text
+/post_hoc_evaluator@post_hoc_evaluator.evaluators.prediction.mauve=mauve_text
```

**Override** packaged fields after composition::

```yaml
post_hoc_evaluator:
  evaluators:
    prediction:
      mauve:
        human_text_source: hf_streaming
```

```yaml
# Or set _target_ explicitly (e.g. task-specific evaluators)
post_hoc_evaluator:
  _target_: xlm.tasks.math500.Math500Eval
```

Example **chaining** MAUVE and generative perplexity on the same prediction loader (recommended: **dict** of sub-evaluators under the pattern key; run order follows key order):

```yaml
post_hoc_evaluator:
  _target_: xlm.tasks.composite_eval.CompositePostHocEvaluator
  evaluators:
    prediction:
      mauve:
        _target_: xlm.tasks.owt.mauve_text_eval.MauveTextEval
        generated_field: text
      gen_ppl:
        _target_: xlm.tasks.owt.generative_perplexity_post_hoc.GenerativePerplexityPostHocEval
        default_judge_device: cuda
        evaluators:
          gpt2-large:
            _target_: xlm.generative_perplexity.AutoModelForCausalLMGenerativePerplexityEvaluator
            name: gpt2-large
            batch_size: 64
```

The same chain can be written as a **list** instead of a dict; dict keys are only for composition and documentation (they are not prepended to logged metric names).

**MAUVE** needs the optional dependency: `pip install "xlm-core[mauve]"` (see [`mauve_text_eval.py`](../../src/xlm/tasks/owt/mauve_text_eval.py)). For unconditional text runs with no per-row reference, override `human_text_source: hf_streaming` under `post_hoc_evaluator.evaluators.prediction.mauve` (or your chosen alias).

Example combining **reported metrics** on a prediction dataloader with **MAUVE** post-hoc:

```yaml
defaults:
  - your_eval_experiment
  - /metrics@reported_metrics.val.math500_prediction.exact_match: seq2seq_exact_match
  - /post_hoc_evaluator@post_hoc_evaluator.evaluators.prediction.mauve: mauve_text

post_hoc_evaluator:
  evaluators:
    prediction:
      mauve:
        human_text_source: null   # default: use per-row reference / truth; set hf_streaming when needed
```

Task authors: wiring a new prediction stream and collator is covered in [Adding a task](adding-a-task.md) (“Wire metrics and optional post-hoc evaluation”).


