# FlexMDM — Flexible Masked Diffusion Model

## Overview

`flexmdm` implements variable-length masked diffusion for text (linear insertion/unmasking noise, seq2seq and unconditional setups). Hydra configs live under {{ gh_dir('xlm-models/flexmdm/configs', 'xlm-models/flexmdm/configs/') }}.

OWT-scale training and eval: [OWT FlexMDM experiment](../experiments/owt_flexmdm.md).

## TinyGSM

Task dataset and preprocessing: [TinyGSM](../tasks/tinygsm.md). GSM8K and code-execution eval: [tinygsm_gsm8k.md](../tasks/tinygsm_gsm8k.md).

| | |
|---|---|
| Experiment | `experiment=tinygsm_flexmdm` |
| Datamodule | {{ gh('xlm-models/flexmdm/configs/datamodule/tinygsm_flexmdm.yaml', 'tinygsm_flexmdm') }} |
| Experiment YAML | {{ gh('xlm-models/flexmdm/configs/experiment/tinygsm_flexmdm.yaml', 'tinygsm_flexmdm') }} |

Register a distinct pad token (`pad_token: "<|pad|>"` in `global_components.tokenizer.special_tokens`) so `pad_token_id != eos_token_id`; otherwise training and prediction will raise at init.

### Training settings

| Setting | Value |
|---------|--------|
| Tokenizer | Qwen2-0.5B (`Qwen/Qwen2-0.5B`) with added `<|mask|>` |
| `block_size` | 512 |
| `input_block_size` | 0 |
| Batching | Per-device 32; global 512 |
| Collators | STAR seq2seq (`seq2seq_*` / `seq2seq_pred_*`); no BOS between question and code |
| Val / test prediction | Post-hoc `code_exec_accuracy` (`Gsm8kCodeEval`); token EM disabled |
| Monitored metric | `val/lm/accumulated_loss` |
| Training schedule | Up to 1M steps; validation every 50k steps; checkpoint every 2.5k steps (keep every 100k) |

Collators are reused from existing STAR seq2seq configs; no TinyGSM-specific collator YAMLs.

### Commands

**Prepare cache** (rank 0 before multi-GPU training):

```bash
xlm job_type=prepare_data experiment=tinygsm_flexmdm num_dataset_workers=8
```

Managers that share `full_name: TinyGSM/TinyGSM/train` use distinct manual cache
directories via `filter_suffix` (e.g. `val_holdout`, `pred_preprocess`). After
changing prediction preprocess, rebuild with
`datamodule.rewrite_manual_cache=true` or delete the stale
`.../TinyGSM/TinyGSM_pred_preprocess/train` tree if needed.

On SLURM, see {{ gh('lib/slurm_scripts/submit_prepare_data.py', 'submit_prepare_data.py') }}.

**Train** (DDP):

```bash
xlm job_name=tinygsm_flexmdm job_type=train experiment=tinygsm_flexmdm \
  per_device_batch_size=32 trainer_strategy=ddp trainer.devices=8 trainer.num_nodes=1 \
  ++trainer.precision=bf16-mixed compile=False
```

### Debug overfit (one TinyGSM train example)

Configs: {{ gh('xlm-models/flexmdm/configs/debug/overfit_tinygsm_flexmdm.yaml', 'debug/overfit_tinygsm_flexmdm') }}, {{ gh('xlm-models/flexmdm/configs/datasets/tinygsm_debug_one.yaml', 'datasets/tinygsm_debug_one') }}, {{ gh('xlm-models/flexmdm/configs/datasets/tinygsm_debug_one_pred.yaml', 'datasets/tinygsm_debug_one_pred') }}.

Train and val/lm share one row (`filter_suffix: debug_one`); val/prediction uses prod `tinygsm_pred_preprocess_fn` on that row (`debug_one_pred`). Prefer this over generic `debug=overfit` for TinyGSM FlexMDM.

**Prepare debug caches** (once; `num_dataset_workers=1` required for the first-row filter):

```bash
xlm job_type=prepare_data experiment=tinygsm_flexmdm debug=overfit_tinygsm_flexmdm \
  datamodule.rewrite_manual_cache=true num_dataset_workers=1
```

**Debug train**:

```bash
xlm job_type=train experiment=tinygsm_flexmdm debug=overfit_tinygsm_flexmdm
```

### Experiment results

Full W&B write-ups under `docs/experiments/` are deferred until runs exist. Use `experiment=tinygsm_flexmdm` with the [document experiment](../guide/eval.md) workflow when ready.
