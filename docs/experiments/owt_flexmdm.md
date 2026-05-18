# OWT FlexMDM

**See also:** [FlexMDM package](../../xlm-models/flexmdm/README.md) · [Task docs](../tasks/owt.md)

## Dataset

Experiment config: [`experiment=owt_flexmdm`](../../xlm-models/flexmdm/configs/experiment/owt_flexmdm.yaml) (datamodule: [`owt_flexmdm`](../../xlm-models/flexmdm/configs/datamodule/owt_flexmdm.yaml)).

Training and validation use [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext), pre-tokenized with GPT-2 and filtered to sequences of at most 1,024 tokens. The processed split is hosted on Hugging Face as [`dhruveshpatel/owt-gpt2-1024-split`](https://huggingface.co/datasets/dhruveshpatel/owt-gpt2-1024-split): a 10k-example validation holdout (seed 2357) and the remainder for training.

| Setting            | Value                                                                                                                |
|--------------------|----------------------------------------------------------------------------------------------------------------------|
| Tokenizer          | GPT-2 (`gpt2`)                                                                                                       |
| Block size         | 1,024                                                                                                                |
| Batching           | Per-device batch size 32; global batch size 512                                                                      |
| Train split        | `dhruveshpatel/owt-gpt2-1024-split/train`                                                                            |
| Val split          | `dhruveshpatel/owt-gpt2-1024-split/validation`                                                                       |
| Train collator     | `FlexMDMTrainCollator` (linear insertion/unmasking noise; variable-length segments truncated to block, EOS appended) |
| Unconditional eval | `FlexMDMEmptyDataset` (`unconditional_prediction` dataloader; empty prompts, max length 1,024)                       |

Training runs for up to 1M steps with validation every 50k steps; checkpoints are saved every 2,500 steps (every 100k steps kept permanently).

## Training

W&B run: [owt_flexmdm](https://wandb.ai/ilm-extensions/LFlexMDM-Text/runs/owt_flexmdm) (`owt_flexmdm`)

```bash
xlm job_name=owt_flexmdm job_type=train experiment=owt_flexmdm \
  per_device_batch_size=32 trainer_strategy=ddp trainer.devices=8 trainer.num_nodes=1 \
  ++trainer.precision=bf16-mixed compile=False \
  +loggers.wandb.resume=allow +loggers.wandb.id=owt_flexmdm
```

## Evaluation

Reference W&B eval run: [owt_flexmdm_eval_step-800000_null_0.95_1024](https://wandb.ai/ilm-extensions/flexmdm/runs/n55k2mel) (`n55k2mel`). Checkpoint is loaded from Hugging Face Hub (`dhruveshpatel/flexmdm-owt`, revision `step-800000`). Gen. PPL uses `experiment=[owt_flexmdm,gpt2_generative_perplexity]` (no MAUVE post-hoc eval).

Single eval (set `HUB_REVISION`, `TOP_P`, and `MAX_STEPS` / sampling budget \(T\); `confidence=null` matches the logged sweep):

```bash
HUB_REVISION=step-800000
TOP_P=0.95
MAX_STEPS=1024
CHECKPOINT_TAG="${HUB_REVISION#step-}"

xlm job_name=owt_flexmdm_eval_${HUB_REVISION}_null_${TOP_P}_${MAX_STEPS} \
  job_type=eval experiment=[owt_flexmdm,gpt2_generative_perplexity] \
  ++eval.checkpoint_path=None ++eval.split=validation \
  per_device_batch_size=16 per_device_val_batch_size=16 global_batch_size=16 \
  trainer_strategy=single_device ++trainer.precision=32-true compile=false \
  loggers=wandb +loggers.wandb.resume=allow +loggers.wandb.id=null \
  ~datamodule.dataset_managers.val.lm \
  +hub.repo_id=dhruveshpatel/flexmdm-owt +hub.revision=${HUB_REVISION} \
  ++predictor.confidence=null ++predictor.top_k=null ++predictor.top_p=${TOP_P} \
  ++predictor.max_steps=${MAX_STEPS} \
  +tags.checkpoint=${CHECKPOINT_TAG} \
  paths.log_dir=logs/eval
```

Reproduce the **Results** table below (\(p=0.95\), \(T \in \{128,256,512,1024\}\), checkpoint `step-800000`):

```bash
HUB_REVISION=step-800000
TOP_P=0.95
CHECKPOINT_TAG="${HUB_REVISION#step-}"

for MAX_STEPS in 128 256 512 1024; do
  xlm job_name=owt_flexmdm_eval_${HUB_REVISION}_null_${TOP_P}_${MAX_STEPS} \
    job_type=eval experiment=[owt_flexmdm,gpt2_generative_perplexity] \
    ++eval.checkpoint_path=None ++eval.split=validation \
    per_device_batch_size=16 per_device_val_batch_size=16 global_batch_size=16 \
    trainer_strategy=single_device ++trainer.precision=32-true compile=false \
    loggers=wandb +loggers.wandb.resume=allow +loggers.wandb.id=null \
    ~datamodule.dataset_managers.val.lm \
    +hub.repo_id=dhruveshpatel/flexmdm-owt +hub.revision=${HUB_REVISION} \
    ++predictor.confidence=null ++predictor.top_k=null ++predictor.top_p=${TOP_P} \
    ++predictor.max_steps=${MAX_STEPS} \
    +tags.checkpoint=${CHECKPOINT_TAG} \
    paths.log_dir=logs/eval
done
```

## Results

Unconditional generation metrics for FlexMDM (variable-length masked diffusion) checkpoints. Gen. PPL is with respect to GPT-2 Large; entropy is the vocabulary entropy of the generated token distribution. Evaluated with nucleus sampling (\(p=0.95\)) for predictor budgets \(T \in \{128, 256, 512, 1024\}\) on 1,000 samples up to 1,024 tokens. MAUVE was not run for this baseline.

| **Checkpoint** | **Gen. PPL (↓)** |       |       |       | **Entropy (↑)** |      |      |      | **MAUVE (↑)** |     |     |      |
|----------------|-----------------:|------:|------:|------:|----------------:|-----:|-----:|-----:|--------------:|----:|----:|-----:|
|                |              128 |   256 |   512 |  1024 |             128 |  256 |  512 | 1024 |           128 | 256 | 512 | 1024 |
| 800k           |            64.68 | 62.08 | 59.61 | 59.27 |            4.93 | 4.88 | 4.88 | 4.92 |             — |   — |   — |    — |

Source: [`notebooks/owt_flexmdm_external_step800k_runs.json`](../../../notebooks/owt_flexmdm_external_step800k_runs.json) (`ilm-extensions/flexmdm`, `tags.checkpoint=800000`, `top_p=0.95`).
