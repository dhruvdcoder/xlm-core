# XLM

XLM is a unified framework for developing and comparing small non-autoregressive language models. It uses PyTorch as the deep learning framework, PyTorch Lightning for training utilities, and Hydra for configuration management.

## Installation

```bash
pip install xlm-core
```

## Usage

```bash
xlm job_type=[JOB_TYPE] job_name=[JOB_NAME] experiment=[CONFIG_PATH]
```

The `job_type` argument can be one of `train`, `eval`, or `generate`. The `experiment` argument should point to the root Hydra config file.

## Example: ILM on LM1B

The following example demonstrates all workflows for the Insertion Language Model (ILM) on the LM1B dataset.

### 1. Prepare Data

```bash
xlm job_type=prepare_data job_name=lm1b_prepare experiment=lm1b_ilm
```

### 2. Train

```bash
# Debug run (overfit on a single batch)
xlm job_type=train job_name=lm1b_ilm experiment=lm1b_ilm debug=overfit

# Full training
xlm job_type=train job_name=lm1b_ilm experiment=lm1b_ilm
```

### 3. Evaluate

```bash
xlm job_type=eval job_name=lm1b_ilm experiment=lm1b_ilm +eval.ckpt_path=<CHECKPOINT_PATH>
```

### 4. Generate

```bash
xlm job_type=generate job_name=lm1b_ilm experiment=lm1b_ilm +generation.ckpt_path=<CHECKPOINT_PATH>
```

Use `debug=[overfit,print_predictions]` to print generated samples to the console:

```bash
xlm job_type=generate job_name=lm1b_ilm experiment=lm1b_ilm +generation.ckpt_path=<CHECKPOINT_PATH> debug=[overfit,print_predictions]
```

### 5. Push to Hub

Upload trained model weights to Hugging Face Hub:

```bash
xlm job_type=push_to_hub job_name=lm1b_ilm_hub experiment=lm1b_ilm +hub_checkpoint_path=<CHECKPOINT_PATH> +hub.repo_id=<REPO_ID>
```
