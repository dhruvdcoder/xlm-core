This page contains information about how to train large models using FSDP.

# Setup

xlm-core trains large models with PyTorch's [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/docs/stable/fsdp.html) via Lightning's [`FSDPStrategy`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html). The configuration is split across **three layers** so that the parts that change rarely (sharding strategy) are separated from the parts that depend on the model (which class to wrap, which dtypes to use):

1. **Base strategy config** (`trainer_strategy/fsdp.yaml`, in xlm-core) — sharding strategy, CPU offload, `use_orig_params`, and `state_dict_type`. Model-agnostic.
2. **Experiment config** — model-specific options: `auto_wrap_policy`, `activation_checkpointing_policy`, `mixed_precision`. Set in your project's `configs/experiment/*.yaml` (e.g. the `dream_fsdp_args.yaml` example below).
3. **CLI / launcher** — selects the strategy at run time with `trainer.strategy=fsdp` (and composes the experiment YAML).

## 1. The base `trainer_strategy/fsdp.yaml`

```1:7:lib/xlm-core/src/xlm/configs/lightning_train/trainer_strategy/fsdp.yaml
# @package trainer.strategy
# Model-specific options (auto_wrap_policy, mixed_precision, etc.) belong in experiment YAML.
_target_: lightning.pytorch.strategies.FSDPStrategy
sharding_strategy: FULL_SHARD
cpu_offload: false
use_orig_params: false
state_dict_type: sharded
```

Notes:

- `sharding_strategy: FULL_SHARD` shards parameters, gradients, and optimizer state across all ranks (ZeRO-3 equivalent). Use `SHARD_GRAD_OP` (ZeRO-2) or `NO_SHARD` (DDP) if memory is not the bottleneck.
- `state_dict_type: sharded` writes one shard per rank instead of consolidating to a single full state dict. This is the only practical option for 7B+ models — a `full` state dict has to materialize the unsharded weights on rank 0, which is what we are trying to avoid in the first place.
- `use_orig_params: false` is the default; flip to `true` only if you need parameter-group-aware optimizers or `torch.compile` over the wrapped model.

## 2. The model-specific experiment YAML (Dream example)

Layered on top of the base strategy, the experiment YAML supplies the three knobs FSDP needs to actually shard a specific model:

```1:13:learned-correctors/dream_correction/configs/experiment/dream_fsdp_args.yaml
# @package _global_
trainer:
  strategy:
    auto_wrap_policy:
      _target_: xlm.utils.fsdp_grouping.make_layer_wrap_policy
      _args_:
        - xlm.backbones.dream.modeling_dream.DreamDecoderLayer
    activation_checkpointing_policy:
      _target_: xlm.utils.fsdp_grouping.make_layer_wrap_policy
      _args_:
        - xlm.backbones.dream.modeling_dream.DreamDecoderLayer
    mixed_precision:
      _target_: xlm.utils.fsdp_grouping.fsdp_bf16_mixed_precision
```

Walking through each block:

### `auto_wrap_policy`

Tells FSDP **which submodule class to treat as a sharding unit**. Each instance of the class becomes its own FSDP unit — its parameters are gathered for the forward, the gradients are reduced/scattered after the backward, and its sharded shard lives on a single rank between steps. For a transformer, this should be the decoder/encoder block class (here, `DreamDecoderLayer`). Wrapping at the block level — not the whole model and not individual `nn.Linear`s — is what gives FSDP its memory savings without flooding the network with tiny collectives.

`xlm.utils.fsdp_grouping.make_layer_wrap_policy` simply imports the dotted class paths you pass and returns the `set` of classes that Lightning's `FSDPStrategy` expects. Pass multiple classes if your model mixes block types:

```python
from xlm.utils.fsdp_grouping import make_layer_wrap_policy

policy = make_layer_wrap_policy(
    "xlm.backbones.dream.modeling_dream.DreamDecoderLayer",
    "xlm.backbones.some_other_model.SomeOtherBlock",
)
```

### `activation_checkpointing_policy`

Selects which submodules get **activation checkpointing** (recompute activations in the backward pass instead of storing them). For Dream-7B at `seq_len=1024`, activation memory dominates, so we re-checkpoint at the same granularity as the FSDP unit. Reusing `make_layer_wrap_policy` keeps the two policies aligned.

If you set `auto_wrap_policy` but not `activation_checkpointing_policy`, you get sharding without recompute — fine for smaller models but typically not enough at 7B.

### `mixed_precision`

```48:57:lib/xlm-core/src/xlm/utils/fsdp_grouping.py
def fsdp_bf16_mixed_precision():
    """Default FSDP mixed precision: bf16 params, fp32 reductions (matches DreamOn reference)."""
    import torch
    from torch.distributed.fsdp import MixedPrecision

    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
```

This is **FSDP-native mixed precision**: parameters are kept in bf16 for compute, the gradient all-reduce / reduce-scatter happens in fp32 to avoid bf16 numerical drift, and buffers (e.g. RoPE caches, attention masks) stay in fp32 because they are read but never reduced. These three dtypes are passed straight to `torch.distributed.fsdp.MixedPrecision` and govern how FSDP stores parameters and runs collectives.

`Trainer(precision="bf16-mixed")` is **complementary, not redundant**, to this block — it controls a different thing. Lightning's `FSDPStrategy.mixed_precision_config` resolves the FSDP `MixedPrecision` like this:

```python
@property
def mixed_precision_config(self):
    if self.mixed_precision is not None:
        return self.mixed_precision      # explicit strategy arg WINS
    plugin = self.precision_plugin
    if isinstance(plugin, FSDPPrecision):
        return plugin.mixed_precision_config
    return None
```

So once you set `mixed_precision` on the strategy, `Trainer(precision=...)` cannot override the FSDP-internal dtypes — the strategy's bf16/fp32/fp32 config is what FSDP actually applies. What `Trainer(precision="bf16-mixed")` *still* does, even when overridden, is wrap the forward in `torch.autocast("cuda", dtype=torch.bfloat16)` via `FSDPPrecision.forward_context`. With `param_dtype=bf16` already, autocast is mostly a no-op (matmuls run in bf16 either way) but it gives a small amount of op-level numerical protection (e.g. cross-entropy intermediates) and matches DreamOn's reference setup, which wraps its forward in an explicit `torch.autocast(bf16)` block.

For a different precision regime (e.g. fp16 with a loss scaler, or pure bf16 with `reduce_dtype=torch.bfloat16`), define your own factory and point `_target_` at it.

## 3. CLI invocation

The base `lightning_train/config.yaml` defaults to `trainer_strategy: single_device`. To switch on FSDP at launch time, add `trainer.strategy=fsdp` and compose your `dream_fsdp_args` overrides in the experiment list:

```bash
xlm \
  job_type=train \
  job_name=opencoder_dream_correction_deletion \
  experiment=[opencoder_dream_correction,dream_fsdp_args] \
  per_device_batch_size=1 trainer.devices=8 trainer.num_nodes=1 \
  trainer.strategy=fsdp compile=false \
  ++trainer.precision=bf16-mixed
```

You may also need to drop callbacks that don't play well with sharded checkpoints during smoke runs:

```bash
~callbacks.checkpoint_monitor ~callbacks.on_exception_checkpoint
```

Three things to call out about this command line:

- `experiment=[opencoder_dream_correction,dream_fsdp_args]` is Hydra list-composition — the second entry **overlays** on the first, so the FSDP strategy block lands inside the existing `trainer:` config rather than replacing it.
- `trainer.strategy=fsdp` is the Hydra group override for `trainer_strategy`. The base YAML uses `# @package trainer.strategy`, so its keys (`sharding_strategy`, `cpu_offload`, …) merge underneath the experiment's `_target_: FSDPStrategy` and the model-specific keys above.
- `++trainer.precision=bf16-mixed` is intentionally on top of the strategy's explicit `mixed_precision` block. It does not change FSDP's parameter/reduction/buffer dtypes (the explicit `mixed_precision` wins; see the section above) — its only effect here is adding the `torch.autocast(bf16)` wrapper around the forward, mirroring DreamOn's reference setup. Drop it if you want a strict no-autocast forward.

## 4. Diagnostics

When FSDP misbehaves, three things go wrong most often: the wrap policy did not match any modules (so the model is not actually sharded), the dtype config did not propagate (so memory is double what you expect), or activation memory dominates a particular phase. The `FSDPDiagnosticsCallback` covers all three.

```15:20:learned-correctors/dream_correction/configs/experiment/dream_fsdp_args.yaml
callbacks:
  fsdp_diagnostics:
    _target_: xlm.utils.fsdp_diagnostics_callback.FSDPDiagnosticsCallback
    num_logged_batches: 3
    log_module_tree_top_k: 5
    log_to_logger: true
```

What it reports (rank 0, prefixed `[FSDPDiagnostics …]` in the log):

1. **Resolved strategy settings** at `setup` — `sharding_strategy`, `cpu_offload`, `use_orig_params`, `auto_wrap_policy`, `activation_checkpointing_policy`, `mixed_precision`, `state_dict_type`, `precision_plugin`, and `precision_plugin.mixed_precision_config`. Use this to confirm the YAML actually merged the way you expected.

   You will normally see **two `MixedPrecision` lines** here when both `mixed_precision` (on the strategy) and `Trainer(precision=...)` are set, and they may print *contradictory* dtypes (e.g. `mixed_precision=MixedPrecision(param_dtype=bf16, reduce_dtype=fp32, ...)` and `precision_plugin.mixed_precision_config=MixedPrecision(param_dtype=fp32, reduce_dtype=bf16, ...)`). This is intentional. The first is what FSDP actually uses; the second is what Lightning *would have* built from `Trainer(precision=...)` if you had not set `mixed_precision`. Per the resolver in `FSDPStrategy.mixed_precision_config`, the explicit one wins and the plugin's value is shadowed.

2. **Post-wrap module tree** at `on_fit_start` — number of `FullyShardedDataParallel` units, number of `CheckpointWrapper`s, sample names, and per-rank `local_trainable_param_MiB`. If `fsdp_units` is `1` or `0`, your `auto_wrap_policy` did not match the layer class (typo, wrong dotted path, or the layer is wrapped by something opaque like `nn.Sequential` your way). Per-rank shard sizes should be roughly `total_params / world_size`. Note the count is per-layer wrappers only — PyTorch always adds one outer "root" FSDP wrap, so the actual `FullyShardedDataParallel` instance count is `fsdp_units + 1`.
3. **Per-phase peak GPU memory** for the first `num_logged_batches` batches — `batch_start`, `after_backward`, `before_optimizer_step`, `batch_end`. Forward-heavy peaks point at activations (more aggressive checkpointing or a smaller `seq_len`); optimizer-step peaks point at parameter / state shards (consider `cpu_offload: true` or smaller `per_device_batch_size`). On large-vocab models the `after_backward` peak is often dominated by the `[B, S, V]` logits and their fp32 gradient, not by activations.

## 5. Checkpointing and resuming

FSDP checkpointing is meaningfully different from the usual single-GPU / DDP path, and the `xlm-core` defaults are tuned for the FSDP variant. The two things to know up front:

1. A "checkpoint" with FSDP is **not necessarily a single `.ckpt` file** — under `state_dict_type: sharded` (the default in `trainer_strategy/fsdp.yaml`) it is a **directory**.
2. Both saving and loading are **collective operations**: every rank must reach the same point at the same time. A failure on one rank during the save can leave the entire process group in a bad state.

### Two `state_dict_type` modes

| Mode | Filesystem layout | Memory at save time | Use case |
|---|---|---|---|
| `sharded` (xlm-core default) | Directory containing one `__N_M.distcp` shard per rank plus a single `meta.pt` written by rank 0. | Each rank writes its own shard; **no rank gathers the full weights**. | Training runs, especially anything that cannot fit the full model on one GPU. |
| `full` | Single `.ckpt` file (rank 0 gathers everything). | Rank 0 must hold full unsharded weights + optimizer state in memory. **Prohibitive at 7B+** in fp32 master copies. | Final export, hub upload, single-GPU eval. |

A sharded-checkpoint directory looks like this on disk:

```text
checkpoints/last.ckpt/
├── meta.pt                    # rank-0 only: trainer/callback state, hyperparameters, global_step, ...
├── __0_0.distcp               # rank 0 model + optimizer shard
├── __1_0.distcp               # rank 1 ...
├── __2_0.distcp
└── ... (one .distcp per rank)
```

Lightning's heuristic for distinguishing the two formats is exactly this:

```python
def _is_sharded_checkpoint(path):
    return path.is_dir() and (path / "meta.pt").is_file()

def _is_full_checkpoint(path):
    return path.is_file()
```

So `last.ckpt` may be a *file* (full) or a *directory* (sharded) depending on the strategy. Anything that uses `os.path.isfile` to detect a checkpoint will silently miss sharded ones — this matters for the auto-resume path below.

### Saving is a collective; the on-exception save can hang

`FSDPStrategy.save_checkpoint` calls `_distributed_checkpoint_save` (sharded) or `FSDP.summon_full_params` + `torch.save` (full). Both require **every rank** to enter the call together. The practical consequences:

- **All ranks must successfully complete validation / training-step before the save fires.** If one rank errors during the train or validation hook, the others will block on the next collective. The save then either hangs until the NCCL watchdog timeout or throws `DistBackendError` — the failure mode the OOM/checkpoint hang debugging session in this repo ran into during validation-triggered checkpointing.
- **`OnExceptionCheckpoint` is risky under FSDP.** When Lightning's exception path triggers a *second* `trainer.save_checkpoint(...)` after a failure, it issues another full set of collectives over a process group that may already be poisoned. The typical symptom is the run flushing one rank-0 INFO line ("Saving checkpoint on exception ...") and then hanging silently until the heartbeat timer fires. **Drop `callbacks.on_exception_checkpoint` for FSDP runs** unless you have a specific reason to keep it; you can do this on the CLI:

  ```bash
  ~callbacks.on_exception_checkpoint
  ```

- **Sharded `ModelCheckpoint` with `save_top_k > 0`** is fine, but expect each save to take longer than the equivalent DDP save: it's a write barrier across the world. Setting `every_n_train_steps` low (e.g. every 100 steps at 7B) can dominate wall-clock time.

### Loading happens *after* FSDP wrap, not before

Two non-obvious flags on `FSDPStrategy` change the lifecycle of resume-from-checkpoint:

- `restore_checkpoint_after_setup = True` — Lightning loads the checkpoint **after** `_setup_model` has wrapped the module in FSDP. This is the opposite of the standard non-FSDP flow, where weights are loaded into the bare `nn.Module` first and the strategy then handles distribution. The wrap must therefore succeed before any weights are read; if your `auto_wrap_policy` is broken, the load also fails. This also means `skip_init_weights: true` is safe to use with FSDP resume — the random-init step is skipped, and FSDP loads the real weights into already-sharded `FlatParameter`s.
- `lightning_restore_optimizer = False` — Lightning's normal optimizer-state-restore code path is *bypassed* under FSDP; FSDP's own `optim_state_dict_to_load` re-flattens the saved optimizer state into the per-rank `FlatParameter` layout. The upshot is that a checkpoint saved at world size W can typically be loaded at world size W' (the `torch.distributed.checkpoint` API handles re-sharding for `sharded` checkpoints), but the model architecture and FSDP wrap policy must match between save and load.

For sharded checkpoints, `load_checkpoint` calls `_distributed_checkpoint_load` (a collective), then loads optimizer state per optimizer via `FSDP.optim_state_dict_to_load`, then `torch.load`s the rank-0 `meta.pt` for the trainer/callback metadata. All of that happens after the wrap, so the very first thing it requires is a healthy NCCL process group.

### Resuming and extracting weights in `xlm-core`

The auto-resume path in `lightning_train.py` is currently **not sharded-aware**:

```99:117:lib/xlm-core/src/xlm/commands/lightning_train.py
    ckpt_path = None
    if cfg.resume_from_checkpoint:
        # determine the checkpoint path
        if cfg.resume_checkpoint_path is not None:
            if os.path.isfile(cfg.resume_checkpoint_path):
                ckpt_path = cfg.resume_checkpoint_path
            else:
                raise ValueError(
                    f"The checkpoint path {cfg.resume_checkpoint_path} is not a file."
                )
        else:
            # look for the "last.ckpt" or "on_exception.ckpt" checkpoint in the checkpointing_dir
            ckpt_path = os.path.join(
                cfg.checkpointing_dir, "on_exception.ckpt"
            )
            if not os.path.isfile(ckpt_path):
                ckpt_path = os.path.join(cfg.checkpointing_dir, "last.ckpt")
            if not os.path.isfile(ckpt_path):
                ckpt_path = None
```

Both branches use `os.path.isfile`, which returns `False` for the directory layout sharded checkpoints produce. So:

- **Auto-pickup silently misses sharded `last.ckpt` directories.** A run that crashes will *not* automatically resume from its sharded `last.ckpt` on relaunch — the script will treat it as "no checkpoint" and start from scratch. Until this is fixed in `lightning_train.py`, the workarounds are:
  1. Pass the directory explicitly via `resume_checkpoint_path=...` *and* relax the `os.path.isfile` check (or replace it with `os.path.exists`).
  2. Run with `state_dict_type: full` for small enough models so `last.ckpt` is a regular file (not viable at 7B+).
- **`xlm job_type=extract_checkpoint`** uses `Harness.from_checkpoint(checkpoint_path=...)`, which expects the standard Lightning `.ckpt` *file* layout. To extract weights from a sharded checkpoint you have to **consolidate to a full checkpoint first**. Two options:
  1. Save a final `state_dict_type: full` checkpoint from the training run (e.g. on the last step, on a smaller model, or after sharding to fewer ranks). This requires enough rank-0 memory for the full weights.
  2. Convert offline using PyTorch's `torch.distributed.checkpoint.format_utils.dcp_to_torch_save`, which reads a `.distcp` directory and writes a single `.pt` containing the consolidated state dict.

The recommended pattern at 7B is: train with `state_dict_type: sharded` (fast saves, no rank-0 memory spike), then run a tiny single-GPU job that loads the last sharded checkpoint with `state_dict_type: full` and saves it once for export.

## 6. Gotchas

A few things that have bitten us in practice:

- **Norm clipping and grad-norm logging under FSDP.** With `FSDPStrategy`, Lightning's built-in `gradient_clip_algorithm: norm` routes through `FSDPPrecision`, which raises `MisconfigurationException` — `torch.nn.utils.clip_grad_norm_()` is wrong for sharded `FlatParameter` gradients. Separately, `Harness.on_before_optimizer_step` uses `lightning.pytorch.utilities.grad_norm(self, ...)`, which only sums gradients visible on *that* rank; under FSDP that is a **local shard norm**, not the true global L2 norm (so W&B can show huge, nearly flat curves that do not match training stability).

  Fix: subclass `Harness`, detect the FSDP root on `self.trainer.strategy.model`, and override **two** hooks. A canonical implementation lives in `learned-correctors` as [`dream_correction.fsdp_harness.FSDPHarness`](../../../../dream_correction/fsdp_harness.py); compose it via `dream_fsdp_args.yaml` (sets `lightning_module._target_` and `trainer.gradient_clip_algorithm: norm`).

  Manually, the same pattern is:

  ```python
  from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

  def configure_gradient_clipping(
      self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
  ):
      if gradient_clip_val is None:
          return
      root = self.trainer.strategy.model
      if isinstance(root, FSDP):
          root.clip_grad_norm_(max_norm=float(gradient_clip_val), norm_type=2.0)
          return
      return super().configure_gradient_clipping(
          optimizer,
          gradient_clip_val=gradient_clip_val,
          gradient_clip_algorithm=gradient_clip_algorithm,
      )
  ```

  ```python
  import torch
  import torch.distributed as dist
  from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

  def on_before_optimizer_step(self, optimizer):
      root = self.trainer.strategy.model
      if not isinstance(root, FSDP):
          return super().on_before_optimizer_step(optimizer)

      local_sq = torch.zeros((), device=self.device, dtype=torch.float32)
      for p in root.parameters():
          if p.grad is not None:
              local_sq += p.grad.detach().float().pow(2).sum()
      if dist.is_available() and dist.is_initialized():
          dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
      global_norm = local_sq.sqrt()
      self.log(
          "Total gradient (norm)",
          global_norm,
          on_step=True,
          on_epoch=False,
          prog_bar=False,
          sync_dist=False,
          rank_zero_only=True,
          logger=True,
          add_dataloader_idx=False,
      )
  ```

  With these overrides, `trainer.gradient_clip_algorithm: norm` is the right choice; `gradient_clip_algorithm: value` remains fine for **non-FSDP** experiments (e.g. single-GPU debug) that still use base `Harness`.

- **Don't double-register the model on the predictor.** In `Harness.instantiate_predictor`, the model is attached to the predictor via `object.__setattr__(self.predictor, "model", self.model)` rather than plain `=`. A normal assignment would register the same `nn.Module` as a submodule of both the harness and the predictor, and FSDP would walk those `FlatParameters` twice — roughly doubling GPU memory usage. If you write your own predictor, copy this pattern.

- **`trainer.precision` does not override the strategy's `mixed_precision`.** A common worry is that setting both `mixed_precision` on `FSDPStrategy` and `Trainer(precision="bf16-mixed")` will conflict or double-cast. It does not. The two control different things: the strategy's `mixed_precision` is what FSDP actually uses for parameter storage and collectives (`FSDPStrategy.mixed_precision_config` short-circuits to it before consulting the precision plugin), while `Trainer(precision="bf16-mixed")` only adds the `torch.autocast(bf16)` wrapper around the forward via `FSDPPrecision.forward_context`. Picking between them:
    - `Trainer(precision="bf16-mixed")` (default for new FSDP runs in xlm-core): autocast on, matches DreamOn's reference setup, gives op-level numerical protection (e.g. cross-entropy upcasts intermediates to fp32 inside autocast). Near-no-op cost when params are already bf16.
    - `Trainer(precision="32-true")` (the Lightning default): no autocast. Slightly more memory-friendly because there are no fp32 intermediates from autocast, but loses the numerical-stability cushion. Pick this only if you have a reason.

- **Per-rank logs.** `FSDPDiagnosticsCallback`'s `on_fit_start` log line is per-rank on purpose — every rank prints its own `local_trainable_param_MiB` so you can spot uneven shards. The `setup`-time strategy dump and per-phase memory logs are rank-0 only.

- **Loading large checkpoints.** Combine FSDP with `skip_init_weights: true` and `init_dtype: bfloat16` (see the [LLM eval notes](../../wiki/LLMs.md#initializing-large-models)) so the model never spends time on random init or fp32 materialization before FSDP shards it.
