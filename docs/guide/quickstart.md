# Quick Start

## Installation

```bash
pip install xlm-core
```

For existing model implementations, also install:

```bash
pip install xlm-models
```

## CLI Usage

XLM uses a simple CLI with three main arguments:

```bash
xlm job_type=<JOB> job_name=<NAME> experiment=<CONFIG>
```

| Argument     | Description                                           |
|--------------|-------------------------------------------------------|
| `job_type`   | One of `prepare_data`, `train`, `eval`, or `generate` |
| `job_name`   | A descriptive name for your run                       |
| `experiment` | Path to your Hydra experiment config                  |

## Example: ILM on LM1B

A complete workflow demonstrating the Insertion Language Model on the LM1B dataset:

### 1. Prepare Data

```bash
xlm job_type=prepare_data job_name=lm1b_prepare experiment=lm1b_ilm
```

### 2. Train

```bash
# Quick debug run (overfit a single batch)
xlm job_type=train job_name=lm1b_ilm experiment=lm1b_ilm debug=overfit

# Full training
xlm job_type=train job_name=lm1b_ilm experiment=lm1b_ilm
```

### 3. Evaluate

```bash
xlm job_type=eval job_name=lm1b_ilm experiment=lm1b_ilm \
    +eval.ckpt_path=<CHECKPOINT_PATH>
```

### 4. Generate

```bash
xlm job_type=generate job_name=lm1b_ilm experiment=lm1b_ilm \
    +generation.ckpt_path=<CHECKPOINT_PATH>
```

**Tip:** Add `debug=[overfit,print_predictions]` to print generated samples to the console:

```bash
xlm job_type=generate job_name=lm1b_ilm experiment=lm1b_ilm \
    +generation.ckpt_path=<CHECKPOINT_PATH> \
    debug=[overfit,print_predictions]
```

### 5. Push to Hugging Face Hub

```bash
xlm job_type=push_to_hub job_name=lm1b_ilm_hub experiment=lm1b_ilm \
    +hub_checkpoint_path=<CHECKPOINT_PATH> \
    +hub.repo_id=<YOUR_REPO_ID>
```
