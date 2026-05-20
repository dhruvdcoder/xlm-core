# TinyGSM

[TinyGSM/TinyGSM](https://huggingface.co/datasets/TinyGSM/TinyGSM) on Hugging Face provides roughly 11.8M training examples: math word problems (`question`) and Python solutions (`code`). In xlm-core each example is wired as a **seq2seq MDM** task: the prefix is `question + "\n"` and the suffix is `code`, following the field layout in [PUMA `tiny_gsm.py`](https://github.com/JaeyeonKim01/PUMA/blob/main/data/tiny_gsm.py).

**Memmap pretokenization is not supported** (`labels.bin`, `prompt_mask.bin`, and related offline paths). Data flows only through `DatasetManager`, `job_type=prepare_data`, and iterable shards at train time.

See also: [Adding a task or dataset](../guide/adding-a-task.md).

## Preprocessing

| Step | Detail |
|------|--------|
| Task module | {{ gh('src/xlm/tasks/tinygsm/__init__.py', 'xlm.tasks.tinygsm.tinygsm_preprocess_fn') }} |
| Outputs | `prompt_token_ids` (question + separator), `input_token_ids` (code) |
| On-the-fly processor | `xlm.datamodule.token_ids_to_input_ids_and_prompt_ids` |
| Val split | 5% holdout via `train_test_split` with `seed: 2025`, `size: 0.05` on the HF `train` split |

## Hydra configs (`src/xlm`)

| Config | Path |
|--------|------|
| Base dataset | {{ gh('src/xlm/configs/lightning_train/datasets/tinygsm.yaml', 'datasets/tinygsm.yaml') }} |
| Train | {{ gh('src/xlm/configs/lightning_train/datasets/tinygsm_train.yaml', 'datasets/tinygsm_train.yaml') }} |
| Val (loss) | {{ gh('src/xlm/configs/lightning_train/datasets/tinygsm_val.yaml', 'datasets/tinygsm_val.yaml') }} |
| Val (prediction) | {{ gh('src/xlm/configs/lightning_train/datasets/tinygsm_val_pred.yaml', 'datasets/tinygsm_val_pred.yaml') }} |
| Datamodule skeleton | {{ gh('src/xlm/configs/lightning_train/datamodule/tinygsm.yaml', 'datamodule/tinygsm.yaml') }} |

## Model experiments

| Model | Experiment | Datamodule |
|-------|------------|------------|
| FlexMDM | `experiment=tinygsm_flexmdm` | {{ gh('xlm-models/flexmdm/configs/datamodule/tinygsm_flexmdm.yaml', 'tinygsm_flexmdm') }} |
| MLM | `experiment=tinygsm_mlm` | {{ gh('xlm-models/mlm/configs/datamodule/tinygsm_mlm.yaml', 'tinygsm_mlm') }} |
| ARLM | `experiment=tinygsm_arlm` | {{ gh('xlm-models/arlm/configs/datamodule/tinygsm_arlm.yaml', 'tinygsm_arlm') }} |

Experiment YAMLs: {{ gh('xlm-models/flexmdm/configs/experiment/tinygsm_flexmdm.yaml', 'tinygsm_flexmdm') }}, {{ gh('xlm-models/mlm/configs/experiment/tinygsm_mlm.yaml', 'tinygsm_mlm') }}, {{ gh('xlm-models/arlm/configs/experiment/tinygsm_arlm.yaml', 'tinygsm_arlm') }}.

### Shared training settings

| Setting | Value |
|---------|--------|
| Tokenizer | Qwen2-0.5B (`Qwen/Qwen2-0.5B`) with added `<|mask|>` |
| `block_size` | 512 |
| `input_block_size` | 0 |
| Batching | Per-device 32; global 512 |
| Collators | STAR seq2seq (`seq2seq_*` / `seq2seq_pred_*`); no BOS between question and code |
| Val prediction metrics | `exact_match`, `token_accuracy` (seq2seq model types) |
| Training schedule | Up to 1M steps; validation every 50k steps; checkpoint every 2.5k steps (keep every 100k) |

Collators are reused from existing STAR seq2seq configs; no TinyGSM-specific collator YAMLs.

## Commands

### Prepare cache

Run on rank 0 before multi-GPU training (tokenizes and writes manual cache):

```bash
xlm job_type=prepare_data experiment=tinygsm_flexmdm num_dataset_workers=8
# or: experiment=tinygsm_mlm / experiment=tinygsm_arlm
```

On SLURM, see {{ gh('lib/slurm_scripts/submit_prepare_data.py', 'submit_prepare_data.py') }}.

### Train (OWT-scale DDP)

```bash
# FlexMDM
xlm job_name=tinygsm_flexmdm job_type=train experiment=tinygsm_flexmdm \
  per_device_batch_size=32 trainer_strategy=ddp trainer.devices=8 trainer.num_nodes=1 \
  ++trainer.precision=bf16-mixed compile=False

# MLM
xlm job_name=tinygsm_mlm job_type=train experiment=tinygsm_mlm \
  per_device_batch_size=32 trainer_strategy=ddp trainer.devices=8 trainer.num_nodes=1 \
  ++trainer.precision=bf16-mixed compile=False

# ARLM
xlm job_name=tinygsm_arlm job_type=train experiment=tinygsm_arlm \
  per_device_batch_size=32 trainer_strategy=ddp trainer.devices=8 trainer.num_nodes=1 \
  ++trainer.precision=bf16-mixed compile=False
```

### Debug

Set `DEBUG_OVERFIT=1` to use `full_name_debug` (same HF split; useful for short smoke runs).

## Experiment result pages

Full W&B training/eval write-ups under `docs/experiments/` are deferred until runs exist. Use the experiment names above with the [document experiment](../guide/eval.md) workflow when ready.
