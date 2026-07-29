# Interactive demo (`job_type=demo`)

Interactive prompt → generation for checking that a checkpoint or Hub revision loads and decodes correctly. Lighter than `job_type=eval`: no Trainer validation loop or post-hoc metrics. Decoding uses the experiment predictor’s `generate` method.

Run:

```bash
python -m xlm.commands.cli_demo job_type=demo job_name=<NAME> experiment=<EXPERIMENT> ...
```

Implementation: {{ gh('src/xlm/commands/cli_demo.py', 'cli_demo.py') }}.

## Loading weights

Uses the same resolution path as `job_type=generate` (`config_prefix=generation`):

1. `generation.ckpt_path` / `generation.checkpoint_path` (full Lightning `.ckpt`)
2. `generation.model_only_checkpoint_path`
3. **`+hub.repo_id`** and optional **`+hub.revision`**

Omit local checkpoint overrides when loading from the Hub. Point **`paths.output_dir`** at a fresh directory so an existing `best.ckpt` / `last.ckpt` under `checkpointing_dir` does not take precedence over Hub weights.

Set **`HF_HUB_KEY`** (or `.secrets.env`) for private Hub repos.

Needs a GPU. Prefer **`compile=false`**. At the prompt, enter text to generate a continuation, or `exit` to quit. Non-empty prompts are prefix-conditioned.

## OWT ILM from Hub

Repo: [`dhruveshpatel/ilm-owt`](https://huggingface.co/dhruveshpatel/ilm-owt), revision `step-800000`.

```bash
python -m xlm.commands.cli_demo \
  job_type=demo \
  job_name=owt_ilm_hub_demo \
  experiment=owt_ilm \
  +hub.repo_id=dhruveshpatel/ilm-owt \
  +hub.revision=step-800000 \
  +trainer.precision=32-true \
  compile=false \
  model.force_flash_attn=false \
  predictor.stopping_threshold=0.9 \
  predictor.max_steps=1024 \
  paths.output_dir=/tmp/xlm_cli_demo_ilm_owt
```

Generation stops when the stopping classifier fires or `predictor.max_steps` is reached.

## OWT FlexMDM from Hub

Repo: [`dhruveshpatel/flexmdm-owt`](https://huggingface.co/dhruveshpatel/flexmdm-owt), revision `step-800000`.

```bash
python -m xlm.commands.cli_demo \
  job_type=demo \
  job_name=owt_flexmdm_hub_demo \
  experiment=owt_flexmdm \
  +hub.repo_id=dhruveshpatel/flexmdm-owt \
  +hub.revision=step-800000 \
  +trainer.precision=32-true \
  compile=false \
  model.force_flash_attn=false \
  predictor.max_steps=1024 \
  ++predictor.top_p=0.95 \
  paths.output_dir=/tmp/xlm_cli_demo_flexmdm_owt
```

FlexMDM runs for `predictor.max_steps` diffusion steps. `++predictor.top_p=0.95` matches the OWT FlexMDM eval sampling setting.

## See also

- Batch generation: `job_type=generate` ([Quick Start](quickstart.md))
- Eval with metrics: [Evaluate](eval.md)
