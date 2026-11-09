# TinyGSM training and GSM8K evaluation

TinyGSM models are trained on `TinyGSM/TinyGSM` with question + Python code
(seq2seq MDM). **Validation and test prediction** use code-execution accuracy
(PUMA-style), not token exact match.

## Code

- Training preprocess: `xlm.tasks.tinygsm.tinygsm_preprocess_fn`
- Val/test prediction preprocess: `xlm.tasks.tinygsm.tinygsm_pred_preprocess_fn`
- GSM8K preprocess: `xlm.tasks.tinygsm.gsm8k_preprocess_fn`
- Post-hoc eval: `xlm.tasks.tinygsm.Gsm8kCodeEval` (metric: `code_exec_accuracy`)

Reference: [PUMA gsm8k_eval.py](https://github.com/JaeyeonKim01/PUMA/blob/main/eval/gsm8k_eval.py)

## Val prediction (`val.prediction`)

`tinygsm_val_pred` uses `tinygsm_pred_preprocess_fn`:

- Prefix: question + `\n`
- Suffix at inference: empty (model generates code)
- `answer`: numeric gold from executing reference `code` once at preprocess time

Post-hoc runs at validation epoch end → `val/prediction/code_exec_accuracy`.

Prediction JSONL rows include:

- `text` — full decoded sequence (question prefix + generated suffix)
- `generated_text` — suffix only (`fixed==0` region); used by `Gsm8kCodeEval`

## GSM8K test (`test.gsm8k_prediction`)

`gsm8k_test_pred` uses `gsm8k_preprocess_fn` (numeric gold from `####` in the
GSM8K `answer` field). Same `Gsm8kCodeEval` via dataloader name substring
`prediction`.

## Gemma-USDM

`experiment=tinygsm_gemma_usdm` trains the full 12-layer Gemma-USDM model on the same TinyGSM seq2seq split.
The task metric is `code_exec_accuracy` from `Gsm8kCodeEval`, not token exact match.
Validation is capped at 50 batches in the experiment config.
GSM8K test scoring uses the same evaluator on `test.gsm8k_prediction`.

```bash
xlm job_type=eval job_name=tinygsm_gemma_usdm_gsm8k experiment=tinygsm_gemma_usdm \
  +eval.ckpt_path=/path/to/checkpoint.ckpt
```

## Standalone GSM8K eval (FlexMDM)

See also [FlexMDM — TinyGSM](../models/flexmdm.md#tinygsm) for training and checkpoint prep.

```bash
conda activate /scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core/.venv_xlm_core
cd /scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core/xlm-models/flexmdm

xlm job_type=eval job_name=tinygsm_gsm8k experiment=tinygsm_flexmdm_gsm8k_eval \
  +eval.ckpt_path=/path/to/checkpoint.ckpt
```

## Unit tests

```bash
pytest tests/tasks/test_tinygsm_gsm8k.py
```
