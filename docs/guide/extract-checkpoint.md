A lightning checkpoint contains the model weights, optimizer state, and other training artifacts. You can extract the model weights from a checkpoint using the `xlm job_type=extract_checkpoint` script.

## Single-file `.ckpt` (DDP or FSDP `state_dict_type: full`)

Writes a PyTorch `.pth` via `torch.save` (or pushes to the Hub using the Harness / `PyTorchModelHubMixin`). EMA is supported via `apply_ema`.

!!! example
```bash
xlm "job_type=extract_checkpoint" \
"job_name=owt_flexmdm" \
"experiment=owt_flexmdm" \
+post_training=default \
+post_training.checkpoint_path=location/of/checkpoints/34-400000.ckpt \
+post_training.model_state_dict_path=output/folder/model_state_dict.pth
```

## FSDP sharded directory (`*.distcp` + `meta.pt`)

When `post_training.checkpoint_path` points at a **directory** (Lightning `state_dict_type: sharded` layout), `extract_checkpoint` consolidates to **model-only safetensors** using [`consolidate_model_checkpoint`](../../src/xlm/utils/consolidate_model_checkpoint.py) instead of `Harness.from_checkpoint`.

- Set **`apply_ema=false`**. EMA from sharded checkpoints is **not** supported on this path (use a single-file full checkpoint and the branch above if you need `apply_ema`).
- Optional **`post_training.max_shard_size`** (e.g. `5GB`): HF-style multi-file `.safetensors`; then `model_state_dict_path` must be a **directory** (see consolidate helper).
- Hub push: loads weights via `load_model_for_inference` and publishes a **single** `model.safetensors` through the Harness mixin. For **sharded safetensors uploads** to the Hub, use `job_type=consolidate_checkpoint` with `consolidate_checkpoint.hub.repo_id` instead.

!!! example "Sharded checkpoint to local safetensors"
```bash
xlm job_type=extract_checkpoint \
  experiment=[opencoder_dream_correction,dream_fsdp_args] \
  +post_training=default \
  +post_training.checkpoint_path=/path/to/last.ckpt \
  +post_training.apply_ema=false \
  +post_training.model_state_dict_path=/path/to/model.safetensors
```

See also [serialization.md](serialization.md) and [llms.md §5 checkpointing](llms.md#5-checkpointing-and-resuming).
