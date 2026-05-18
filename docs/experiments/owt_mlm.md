# OWT MLM

**See also:** [Model docs](../models/mlm.md) · [Task docs](../tasks/owt.md)

## Dataset

Experiment config: [`experiment=owt_mlm`](../../xlm-models/mlm/configs/experiment/owt_mlm.yaml) (datamodule: [`owt_mlm`](../../xlm-models/mlm/configs/datamodule/owt_mlm.yaml)).

Training and validation use [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext), pre-tokenized with GPT-2 and filtered to sequences of at most 1,024 tokens. The processed split is hosted on Hugging Face as [`dhruveshpatel/owt-gpt2-1024-split`](https://huggingface.co/datasets/dhruveshpatel/owt-gpt2-1024-split): a 10k-example validation holdout (seed 2357) and the remainder for training.

| Setting            | Value                                                                                      |
|--------------------|--------------------------------------------------------------------------------------------|
| Tokenizer          | GPT-2 (`gpt2`)                                                                             |
| Block size         | 1,024                                                                                      |
| Batching           | Per-device batch size 32; global batch size 512                                            |
| Train split        | `dhruveshpatel/owt-gpt2-1024-split/train`                                                  |
| Val split          | `dhruveshpatel/owt-gpt2-1024-split/validation`                                             |
| Train collator     | `DefaultMLMCollator` (random MLM masking, truncate to block, EOS appended)                 |
| Unconditional eval | `MLMEmptyDataset` (`unconditional_prediction` dataloader; empty prompts, max length 1,024) |

Training runs for up to 1M steps with validation every 50k steps; checkpoints are saved every 2,500 steps (every 100k steps kept permanently).

## Training

W&B run: [owt_mlm](https://wandb.ai/ilm-extensions/idlm2/runs/owt_mlm_v1) (`owt_mlm_v1`)

```bash
xlm job_name=owt_mlm job_type=train experiment=owt_mlm \
  per_device_batch_size=32 trainer_strategy=ddp trainer.devices=8 trainer.num_nodes=1 \
  ++trainer.precision=bf16-mixed compile=true loggers=wandb \
  +loggers.wandb.resume=allow +loggers.wandb.id=owt_mlm_v1
```

## Evaluation

 Checkpoint is loaded from Hugging Face Hub (`dhruveshpatel/mlm-owt`, revision `step-800000`); set `+hub.repo_id` / `+hub.revision` for other checkpoints.

Single eval (set `HUB_REVISION`, `TOP_P`, and `MAX_STEPS` / sampling budget \(T\)):

```bash
HUB_REVISION=step-800000
TOP_P=0.95
MAX_STEPS=1024
STEP="${HUB_REVISION#step-}"

xlm job_name=owt_mlm_eval_${HUB_REVISION}_${TOP_P}_${MAX_STEPS} \
  job_type=eval experiment=owt_mlm \
  ++eval.checkpoint_path=None ++eval.split=validation \
  trainer_strategy=single_device ++trainer.precision=32-true compile=false \
  loggers=wandb +loggers.wandb.resume=allow +loggers.wandb.id=null \
  ~datamodule.dataset_managers.val.lm \
  +hub.repo_id=dhruveshpatel/mlm-owt +hub.revision=${HUB_REVISION} \
  +post_hoc_evaluator@post_hoc_evaluator.evaluators.prediction.gen_ppl=gen_ppl_gpt2_large \
  +post_hoc_evaluator@post_hoc_evaluator.evaluators.prediction.mauve=mauve_text \
  post_hoc_evaluator.evaluators.prediction.mauve.human_text_source=hf_streaming \
  ++predictor.top_p=${TOP_P} ++predictor.top_k=null predictor.max_steps=${MAX_STEPS} \
  +tags.checkpoint=${HUB_REVISION} +tags.step=${STEP} +tags.eval_type=nll \
  paths.log_dir=logs/eval \
  datamodule.dataset_managers.val.unconditional_prediction.num_examples=1000 \
  model.force_flash_attn=false \
  per_device_batch_size=16 global_batch_size=16
```

Reproduce the **Results** table below (\(p=0.95\), \(T \in \{128,256,512,1024\}\), checkpoint `step-800000`):

```bash
HUB_REVISION=step-800000
TOP_P=0.95
STEP="${HUB_REVISION#step-}"

for MAX_STEPS in 128 256 512 1024; do
  xlm job_name=owt_mlm_eval_${HUB_REVISION}_${TOP_P}_${MAX_STEPS} \
    job_type=eval experiment=owt_mlm \
    ++eval.checkpoint_path=None ++eval.split=validation \
    trainer_strategy=single_device ++trainer.precision=32-true compile=false \
    loggers=wandb +loggers.wandb.resume=allow +loggers.wandb.id=null \
    ~datamodule.dataset_managers.val.lm \
    +hub.repo_id=dhruveshpatel/mlm-owt +hub.revision=${HUB_REVISION} \
    +post_hoc_evaluator@post_hoc_evaluator.evaluators.prediction.gen_ppl=gen_ppl_gpt2_large \
    +post_hoc_evaluator@post_hoc_evaluator.evaluators.prediction.mauve=mauve_text \
    post_hoc_evaluator.evaluators.prediction.mauve.human_text_source=hf_streaming \
    ++predictor.top_p=${TOP_P} ++predictor.top_k=null predictor.max_steps=${MAX_STEPS} \
    +tags.checkpoint=${HUB_REVISION} +tags.step=${STEP} +tags.eval_type=nll \
    paths.log_dir=logs/eval \
    datamodule.dataset_managers.val.unconditional_prediction.num_examples=1000 \
    model.force_flash_attn=false \
    per_device_batch_size=16 global_batch_size=16
done
```

## Results

Unconditional generation metrics for MLM (masked diffusion) checkpoints. Gen. PPL is with respect to GPT-2 Large; entropy is the vocabulary entropy of the generated token distribution. Evaluated with nucleus sampling (\(p=0.95\)) for predictor budgets \(T \in \{128, 256, 512, 1024\}\) on 1,000 samples up to 1,024 tokens.

| **Checkpoint** | **Gen. PPL (↓)** |       |       |       | **Entropy (↑)** |      |      |      | **MAUVE (↑)** |      |      |      |
|----------------|-----------------:|------:|------:|------:|----------------:|-----:|-----:|-----:|--------------:|-----:|-----:|-----:|
|                |              128 |   256 |   512 |  1024 |             128 |  256 |  512 | 1024 |           128 |  256 |  512 | 1024 |
| 800k           |            78.34 | 63.27 | 55.82 | 54.25 |            4.09 | 4.48 | 4.70 | 4.86 |          0.39 | 0.68 | 0.76 | 0.86 |

Source: [`notebooks/owt_mlm_step800k_runs.json`](../../../notebooks/owt_mlm_step800k_runs.json) (`rl-discrete-diffusion/correctors`, `tags.checkpoint=step-800000`, `top_p=0.95`).
