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
