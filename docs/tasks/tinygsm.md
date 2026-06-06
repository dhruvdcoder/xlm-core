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
| Val (prediction) | {{ gh('src/xlm/configs/lightning_train/datasets/tinygsm_val_pred.yaml', 'datasets/tinygsm_val_pred.yaml') }} — `tinygsm_pred_preprocess_fn`, code-exec post-hoc |
| GSM8K test | {{ gh('src/xlm/configs/lightning_train/datasets/gsm8k_test_pred.yaml', 'datasets/gsm8k_test_pred.yaml') }} |
| Datamodule skeleton | {{ gh('src/xlm/configs/lightning_train/datamodule/tinygsm.yaml', 'datamodule/tinygsm.yaml') }} |

GSM8K and code-execution eval: [tinygsm_gsm8k.md](tinygsm_gsm8k.md).

## Model experiments

Training settings, prepare/train commands, and experiment YAMLs live in the per-model docs:

| Model | Experiment | Doc |
|-------|------------|-----|
| FlexMDM | `experiment=tinygsm_flexmdm` | [FlexMDM — TinyGSM](../models/flexmdm.md#tinygsm) (debug: `debug=overfit_tinygsm_flexmdm`) |
| MLM | `experiment=tinygsm_mlm` | [MLM — TinyGSM](../models/mlm.md#tinygsm) |
| ARLM | `experiment=tinygsm_arlm` | [ARLM — TinyGSM](../models/arlm.md#tinygsm) |
