# Evaluate a model

You have three options to load a model for evaluation:
1. From a full Lightning checkpoint (`.ckpt`). This checkpoint contains the model weights, optimizer state, and other training artifacts.
2. From a model-only checkpoint (`.pt`). This checkpoint contains only the model weights. A model only checkpoint can be extracted from a full Lightning checkpoint using the `xlm job_type=extract_checkpoint` script.

```bash
xlm job_type=eval job_name=lm1b_ilm_hub experiment=lm1b_ilm \
    +hub_checkpoint_path=<CHECKPOINT_PATH> \
    +hub.repo_id=<YOUR_REPO_ID>
```
