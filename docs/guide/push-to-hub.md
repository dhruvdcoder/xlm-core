# Push to Hugging Face Hub

The `push_to_hub` command uploads a trained XLM model to the [Hugging Face Hub](https://huggingface.co/models), making it easy to share and deploy your model. When loading from a full Lightning checkpoint, **EMA weights are automatically applied** to the model before pushing.

## Prerequisites

1. **Hugging Face account** — Create one at [huggingface.co](https://huggingface.co/join).

2. **Authentication** — Set your Hugging Face token:
   - Create a token at [Settings → Access Tokens](https://huggingface.co/settings/tokens) (with write access).
   - Set the `HF_HUB_KEY` environment variable:
     ```bash
     export HF_HUB_KEY="hf_..."
     ```
   - Or place it in `.secrets.env` (loaded automatically by the command).

## Usage

### Basic usage with full checkpoint

Load from a Lightning checkpoint and push to the Hub:

```bash
xlm job_type=push_to_hub job_name=lm1b_ilm_hub experiment=lm1b_ilm \
    +hub_checkpoint_path=<CHECKPOINT_PATH> \
    +hub.repo_id=<YOUR_REPO_ID>
```

Example:

```bash
xlm job_type=push_to_hub job_name=owt_ilm_hub experiment=owt_ilm \
    +hub_checkpoint_path=logs/owt_ilm/checkpoints/epoch=40-step=422500.ckpt \
    +hub.repo_id=username/my-ilm-model
```

### Using model-only checkpoint

If you have a model-only checkpoint (e.g., from `extract_model_state_dict` or similar), you can push it directly. **Ensure the weights were saved with EMA applied** if you trained with EMA:

```bash
xlm job_type=push_to_hub job_name=my_model_hub experiment=my_experiment \
    +model_only_checkpoint_path=<MODEL_STATE_DICT_PATH> \
    +hub.repo_id=<YOUR_REPO_ID>
```

!!! note "EMA weights"
    When using `model_only_checkpoint_path`, the checkpoint should already contain EMA weights if you want to publish the EMA-averaged model. The `push_to_hub` command does not apply EMA in this case—it loads the weights as-is.

### Custom commit message

Override the default commit message:

```bash
xlm job_type=push_to_hub job_name=my_model_hub experiment=my_experiment \
    +hub_checkpoint_path=<CHECKPOINT_PATH> \
    +hub.repo_id=<YOUR_REPO_ID> \
    +hub.commit_message="Trained on LM1B, 50k steps"
```

### Pushing to a specific branch

Push to a named branch (e.g., to track checkpoint steps). The branch is created automatically if it does not exist:

```bash
xlm job_type=push_to_hub job_name=my_model_hub experiment=my_experiment \
    +hub_checkpoint_path=<CHECKPOINT_PATH> \
    +hub.repo_id=<YOUR_REPO_ID> \
    +hub.branch=step-400000
```

## Configuration reference

| Parameter                    | Required     | Description                                                                                             |
|------------------------------|--------------|---------------------------------------------------------------------------------------------------------|
| `hub_checkpoint_path`        | One of these | Path to a full Lightning checkpoint (`.ckpt`). EMA weights in the checkpoint are automatically applied. |
| `model_only_checkpoint_path` | One of these | Path to a model-only state dict. Loaded as-is; ensure EMA was applied when saved if needed.             |
| `hub.repo_id`                | Yes          | Hugging Face repository ID (e.g., `username/model-name`).                                               |
| `hub.commit_message`         | No           | Custom commit message. Defaults to a message that includes checkpoint paths.                            |
| `hub.branch`                 | No           | Git branch to push to. Defaults to `main`. If the branch does not exist, it is created automatically.   |


# Running eval or generate from a checkpoint on the hub

```bash
xlm job_type=eval job_name=lm1b_ilm_hub experiment=lm1b_ilm \
    +hub_checkpoint_path=<CHECKPOINT_PATH> \
    +hub.repo_id=<YOUR_REPO_ID>
```

```bash
xlm job_type=generate job_name=owt_ilm experiment=owt_ilm \
    +hub.repo_id=dhruveshpatel/ilm-owt \
    +hub.revision=step-400000 \
    +trainer.precision=32-true \
    ... other config overrides
```