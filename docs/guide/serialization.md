# Model serialization and deserialization — status

A complete picture of all the model save / load paths we currently support
(or claim to support) across DDP / single-device and FSDP, both
`state_dict_type: full` (single `.ckpt` file) and
`state_dict_type: sharded` (directory with `*.distcp` + `meta.pt`).

We use this doc as the source of truth for what is **expected to work
out of the box** versus what is **broken / missing** today. The
existing per-feature docs (
[`extract-checkpoint.md`](extract-checkpoint.md),
[`push-to-hub.md`](push-to-hub.md),
{{ gh('wiki/model-loading-for-inference.md', 'model-loading-for-inference.md') }},
[`llms.md`](llms.md))
are partially outdated; the cross-cutting view here is what they
should converge to.

## Concepts and artifacts

There are three on-disk artifacts in this system:

| Name | What it is | Created by | Layout |
|---|---|---|---|
| **Full checkpoint, single-file** | All Lightning state (model + optimizer + EMA + callbacks + trainer/global_step + hyperparameters) in one pickle. | DDP / single-device training. FSDP with `state_dict_type: full` (rank 0 gathers). | `last.ckpt` is a **file**. |
| **Full checkpoint, sharded** | One `*.distcp` shard per rank + a single `meta.pt` written by rank 0 (trainer state, callbacks, EMA, …). | FSDP with `state_dict_type: sharded` (the xlm-core default in `trainer_strategy/fsdp.yaml`). | `last.ckpt/` is a **directory**. |
| **Model-only checkpoint** | Just `model.state_dict()`. Either a single `.safetensors` / `.pt` file, or the HF sharded layout `model.safetensors.index.json` + `model-*-of-*.safetensors`. | `extract_checkpoint`, `consolidate_model_checkpoint`, `Harness.save_model_weights`, `Harness.push_to_hub`. | File, or directory containing the HF index file. |

The two **full** layouts are how training itself writes state; both contain enough
information to resume training. The **model-only** layout is what we publish to
the HF Hub and what we point `model_only_checkpoint_path` (or `hub.repo_id`) at
for inference.

Source-of-truth code:

- Training entry: {{ gh('src/xlm/commands/lightning_train.py', 'xlm/commands/lightning_train.py') }}
- Eval / generate: {{ gh('src/xlm/commands/lightning_eval.py', 'xlm/commands/lightning_eval.py') }}, {{ gh('src/xlm/commands/lightning_generate.py', 'xlm/commands/lightning_generate.py') }}
- Inference loader (used by eval / generate / push_to_hub): {{ gh('src/xlm/utils/model_loading.py', 'xlm/utils/model_loading.py') }} — `load_model_for_inference`
- Extract full → model-only (single-file source): {{ gh('src/xlm/commands/extract_checkpoint.py', 'xlm/commands/extract_checkpoint.py') }}
- Consolidate FSDP sharded → model-only safetensors: {{ gh('src/xlm/utils/consolidate_model_checkpoint.py', 'xlm/utils/consolidate_model_checkpoint.py') }}.
- Resume helpers (file or distcp directory): {{ gh('src/xlm/utils/checkpoint_paths.py', 'xlm/utils/checkpoint_paths.py') }}
- Hub download + safetensors I/O: {{ gh('src/xlm/utils/hf_hub.py', 'xlm/utils/hf_hub.py') }}
- Hub push: {{ gh('src/xlm/commands/push_to_hub.py', 'xlm/commands/push_to_hub.py') }} → `Model.push_to_hub` (`PyTorchModelHubMixin`)
- EMA-aware Harness checkpoint helpers: `Harness.from_checkpoint` / `Harness.save_model_weights` / `Harness.push_to_hub` in {{ gh('src/xlm/harness.py', 'xlm/harness.py') }}

## Status matrix

Legend: ✅ works, ⚠️ works but with caveats (see notes), ❌ broken / not
supported in current code, — not applicable.

Within each workflow, "DDP / single-device" means a single `.ckpt` **file**
(this is also the FSDP `state_dict_type: full` layout). "FSDP sharded"
means a directory with `*.distcp` shards.

### 1. Save full checkpoint during training (Lightning writes it)

| Mode | Layout produced | Status | Notes |
|---|---|---|---|
| DDP / single-device | single `.ckpt` file | ✅ | Lightning's default `ModelCheckpoint`. |
| FSDP `state_dict_type: full` | single `.ckpt` file | ⚠️ | Works; rank 0 gathers full unsharded params + optimizer state. Prohibitive at 7B+. |
| FSDP `state_dict_type: sharded` (xlm-core default) | directory `last.ckpt/` with `*.distcp` + `meta.pt` | ✅ | This is what `trainer_strategy/fsdp.yaml` produces. `OnExceptionCheckpoint` is risky under FSDP — drop it. See [`llms.md` §5](llms.md#5-checkpointing-and-resuming). |

### 2. Resume training from a full checkpoint (`lightning_train.py`)

| Source | Status | Notes |
|---|---|---|
| Explicit `resume_checkpoint_path` = single `.ckpt` file (DDP / FSDP-full) | ✅ | Validated by `resolve_explicit_resume_checkpoint_path`. |
| Explicit `resume_checkpoint_path` = FSDP sharded directory (`*.distcp` + `meta.pt`) | ✅ | Same helper accepts a dir with `*.distcp`. |
| Auto-pickup `on_exception.ckpt` / `last.ckpt` — file | ✅ | `find_auto_resume_checkpoint` accepts a file. |
| Auto-pickup `on_exception.ckpt` / `last.ckpt` — sharded directory | ✅ | Same helper accepts a directory with `*.distcp`. |
| Cross-world-size resume (sharded, W → W′) | ⚠️ | Supported by `torch.distributed.checkpoint` for sharded ckpts; model architecture and FSDP wrap policy must match. Not exercised by tests in this repo. |

Seeding a *new* training run from a pre-trained model-only checkpoint
(no optimizer state, no EMA, no global_step) is a distinct path; see
[§3. Seeding from existing model weights](#3-seeding-from-existing-model-weights-lightning_trainpy)
below.

### 3. Seeding from existing model weights (`lightning_train.py`)

When starting a *new* training run with pre-trained weights,
`lightning_train.py` supports loading **only the model parameters** into
`lightning_module.model` before `trainer.fit`. This is **not** a resume:
no optimizer state, no LR-scheduler state, no `global_step`, no EMA, no
trainer/callback state.

Precedence: if an explicit `resume_checkpoint_path` is set, *or* an
auto-resume hit (`on_exception.ckpt` / `last.ckpt`) is found under
`checkpointing_dir`, that wins and the seeding sources below are
**ignored** (a rank-zero ERROR log is emitted explaining the override).

| Source | DDP / single-device | FSDP (sharded train output) | Notes |
|---|---|---|---|
| `model_only_checkpoint_path` = single `.safetensors` / `.pt` / `.bin` file | ✅ | ✅ | Loaded via `load_model_weights_into_model(map_location="cpu")` into the bare `nn.Module` on every rank; FSDP wrap during `trainer.fit` setup then shards the populated module. |
| `model_only_checkpoint_path` = HF sharded index (`model.safetensors.index.json`) | ✅ | ✅ | `_load_sharded_safetensors_into_model` loads one shard at a time — peak host RAM ≈ model + one shard. This is the only memory-friendly option for FSDP seeding at multi-B scale. |
| `hub.repo_id` (single-file safetensors) | ✅ | ✅ | Every rank calls `download_model_weights`; HF cache file locks deduplicate the actual download. Loading is then identical to the single-file local case. |
| `hub.repo_id` (sharded safetensors) | ✅ | ✅ | `_download_sharded_safetensors` pulls the index + all shards into the HF cache, then loaded one shard at a time. |
| `hub.repo_id` (legacy `pytorch_model.bin`) | ✅ | ✅ | `torch.load` + `load_state_dict`. Single-file path → high peak CPU RAM under FSDP. |
| `strict_model_only_load: false` | ✅ | ✅ | Pass-through to `load_state_dict(strict=False)`. Useful when seeding a head from a base model. Missing / unexpected keys are warned, not raised. |
| `skip_init_weights: true` combined with any of the above | ✅ | ✅ | Module is constructed under `transformers.modeling_utils.no_init_weights()`, then weights are loaded on top. Strongly recommended for FSDP seeding at multi-B scale (saves a full random init on every rank that is about to be overwritten). |

#### FSDP-specific notes on seeding

- **Each rank holds the full state dict briefly on CPU.** Seeding happens
  *before* FSDP wrap, so every rank loads the full weights into its bare
  module. Peak CPU RAM per rank is roughly *model size* (single-file
  source) or *model + one shard* (HF sharded index). At 7B+, prefer the
  HF-sharded source layout — `extract_checkpoint` with `post_training.max_shard_size`
  or a direct call to `consolidate_model_checkpoint` with `max_shard_size` — and
  standard Hub-published models use it.
- **Seeding loads before the wrap; full-checkpoint resume loads after.**
  This is intentional and is the opposite of workflow #2: FSDP sets
  `restore_checkpoint_after_setup = True`, so a *resume* materializes
  weights into already-sharded `FlatParameter`s, but a *seed* fills the
  bare module first and then the strategy shards it. Both arrive at the
  same post-wrap shape; the memory profile and the failure modes differ.
  Practical consequence: an `auto_wrap_policy` mismatch breaks resume
  with a load-time error, but breaks seeding silently at wrap time (the
  module already has weights, FSDP just wraps the whole thing as one
  unit).
- **No EMA on the seeding path.** `load_model_weights_into_model` only
  sets parameters from `state_dict`. If you need EMA-averaged weights as
  the seed, the source file must already contain them
  (`extract_checkpoint apply_ema=True` for single-file ckpts; the
  FSDP-sharded equivalent does not exist today — see workflow #4).
- **`map_location` is hardcoded to `"cpu"`.** Don't point at a CUDA
  device; FSDP wrap is what moves and shards onto GPUs.
- **Conflict with full-checkpoint sources.** If `resume_from_checkpoint`
  is true and either an explicit `resume_checkpoint_path` or an
  auto-resume `on_exception.ckpt` / `last.ckpt` is present, the seeding
  sources are dropped — even if both are explicitly set in the config.
  Rank-zero logs `Resume checkpoint is set; model-only / Hub weight
  sources are ignored.`

### 4. Convert a full checkpoint into a model-only checkpoint

| Source layout | Tool | Output | Status | Notes |
|---|---|---|---|---|
| Single-file `.ckpt` (DDP or FSDP-full) | `xlm job_type=extract_checkpoint` ({{ gh('src/xlm/commands/extract_checkpoint.py', 'extract_checkpoint.py') }}) | `.pth` (`torch.save`) and/or hub push | ✅ | Uses `Harness.from_checkpoint(..., apply_ema=True)` → `Harness.save_model_weights` / `Harness.push_to_hub`. Supports EMA application. |
| FSDP sharded directory | `xlm job_type=extract_checkpoint` | `.safetensors` (and/or Hub as single `model.safetensors`) | ✅ | Dispatches to `consolidate_model_checkpoint`; **`apply_ema` must be false** (raises if true). Hub path uses `load_model_for_inference` → `Harness.push_to_hub`. Optional `post_training.max_shard_size` for HF-sharded local output. |
| FSDP sharded directory | `xlm job_type=extract_checkpoint` ({{ gh('src/xlm/utils/consolidate_model_checkpoint.py', 'consolidate_model_checkpoint') }} via {{ gh('src/xlm/commands/extract_checkpoint.py', 'extract_checkpoint') }}) | single `.safetensors` file, **or** HF sharded layout when `max_shard_size` is set (e.g. `"5GB"`) | ✅ | No EMA on this path (`apply_ema=false`). Optional `post_training.max_shard_size` for HF-sharded local output. Requires enough **CPU RAM** for the full model. |
| FSDP sharded directory **with EMA application** | (none) | — | ❌ | By policy: no EMA on FSDP/sharded extract. Use a single-file full `.ckpt` and `extract_checkpoint` with `apply_ema=true`, or export non-EMA weights from sharded checkpoints. |

The recommended FSDP path at 7B is: **train sharded → consolidate to safetensors
(single or HF-sharded) on a host with enough CPU RAM → publish / load for
inference**.

### 5. Load from local disk for eval / generate (`lightning_eval.py`, `lightning_generate.py`)

Both commands go through `load_model_for_inference` and look at the
`{prefix}.ckpt_path` / `{prefix}.checkpoint_path` keys (full checkpoint) and
the `{prefix}.model_only_checkpoint_path` key (model-only).

| Source | Eval (DDP) | Eval (FSDP, sharded training output) | Generate (single device today) | Notes |
|---|---|---|---|---|
| Single-file `.ckpt` (full) | ✅ | ⚠️ | ✅ | DDP/single-device: `Harness.load_from_checkpoint`. FSDP eval theoretically wraps after weights are loaded into the bare module, but this path is not exercised in tests and **the file-only `os.path.isfile` check on the config value silently rejects sharded directories** (see below). |
| FSDP **sharded directory** as `eval.ckpt_path` / `generation.ckpt_path` | ❌ | ❌ | ❌ | `_get_full_checkpoint_path` in `model_loading.py` does `if not os.path.isfile(ckpt_path): raise ValueError(...)`. Sharded directories are rejected before any FSDP-aware loader runs. **This is the symmetric gap to workflow #2** — train resume accepts dirs, eval/generate does not. |
| Auto-fallback to `best.ckpt` / `last.ckpt` (eval only) | ✅ for file | ❌ for sharded dir | — | `_get_full_checkpoint_path` also uses `os.path.isfile` for the fallback search; sharded `last.ckpt/` directories are never picked up. |
| Local `model_only_checkpoint_path` = single `.safetensors` / `.pt` file | ✅ | ✅ | ✅ | `load_model_weights_into_model` handles `.safetensors` and pickle. For FSDP eval, the bare module is built, weights loaded on CPU, then the trainer wraps and shards on `validate`. |
| Local `model_only_checkpoint_path` = HF sharded layout (point at the `model.safetensors.index.json` file) | ✅ | ✅ | ✅ | `_load_sharded_safetensors_into_model` loads one shard at a time. Peak host RAM ≈ model + one shard. |
| `eval.model_only_checkpoint_path` together with EMA | ⚠️ | ⚠️ | ⚠️ | EMA is *not* re-applied at load time. The file must already contain EMA-averaged weights (`extract_checkpoint` with `apply_ema=true` on single-file ckpts only; **not** on the FSDP-sharded `extract_checkpoint` path — see workflow #4). |

In the inference path, **the `os.path.isfile` checks in `_get_full_checkpoint_path` (both
the explicit and the fallback branches) are the single most impactful bug** — it
makes FSDP-sharded eval / generate / push-to-hub all silently unsupported even when
the rest of the pipeline could handle them.

### 6. Load model-only weights from the HF Hub for eval / generate

| Hub layout | Eval | Generate | Notes |
|---|---|---|---|
| Single-file `model.safetensors` | ✅ | ✅ | `hf_hub.download_model_weights` tries this first. |
| Sharded safetensors (`model.safetensors.index.json` + `model-*-of-*.safetensors`) | ✅ | ✅ | Falls through to `_download_sharded_safetensors`, then `_load_sharded_safetensors_into_model` (per-shard loading). |
| Legacy `pytorch_model.bin` | ✅ | ✅ | Tried last; loaded via `torch.load` + `load_state_dict`. |
| `hub.revision=<branch/tag/commit>` | ✅ | ✅ | Passed through to `hf_hub_download`. |
| `init_dtype: bfloat16` / `float16` + `skip_init_weights: true` (large-model friendly) | ✅ | ✅ | Honored by `load_model_for_inference` — both apply when weights come from a model-only source (Hub or local). They are **not** consulted on the full-checkpoint branch of `load_from_checkpoint` (`init_dtype` is still applied as default dtype, but `skip_init_weights` is ignored). |

FSDP eval reaching for Hub weights is the **same** code path as local
model-only loading — weights are loaded into the bare module on CPU, then
`trainer.validate` wraps and shards. Generate today does not wrap (manual
predict loop), so it is effectively single-device.

### 7. Push to the HF Hub (`push_to_hub.py`)

`push_to_hub` ultimately calls `Model.push_to_hub(...)` (i.e.
`PyTorchModelHubMixin.push_to_hub`), which in turn calls
`PyTorchModelHubMixin._save_pretrained` — that saves
`model.safetensors` (a **single** file) plus generated `config.json` / README,
then uploads the folder.

| Weight source | Status | Notes |
|---|---|---|
| Local single-file `.ckpt` via `hub_checkpoint_path` | ✅ | `load_model_for_inference(config_prefix="")` with `manual_ema_restore=True` → EMA is applied via `Harness.from_checkpoint`. |
| Local FSDP **sharded directory** via `hub_checkpoint_path` | ❌ | Same `_get_full_checkpoint_path` / `os.path.isfile` issue as workflow #5. Consolidate to model-only safetensors first and pass the result via `model_only_checkpoint_path`. |
| Local model-only single file via `model_only_checkpoint_path` (`.safetensors` / `.pt`) | ✅ | Module is instantiated and weights loaded; the upload re-serializes a single `model.safetensors`. EMA is **not** re-applied — make sure the file already has EMA weights. |
| Local model-only HF-sharded layout via `model_only_checkpoint_path=…/model.safetensors.index.json` | ⚠️ | Loading works (per-shard). On upload, `PyTorchModelHubMixin._save_pretrained` only writes a **single** `model.safetensors`, so the multi-shard layout is *flattened*. For ≤ ~50 GB (Hub single-file limit) and enough host RAM that is fine; for larger models the push is unsupported. |
| Push a pre-built multi-shard safetensors folder verbatim | ❌ | No command does this. `push_to_hub` always re-serializes through `_save_pretrained`. You'd have to use `HfApi.upload_folder` yourself. |
| `hub.branch=<name>` | ✅ | Branch is created via `HfApi.create_branch` if missing. |
| `hub.commit_message` defaulting | ✅ | Falls back to a generated message mentioning the source paths. |
| Optional `hub_checkpoint_path` + `model_only_checkpoint_path` both set | ⚠️ | Full ckpt wins; conflict is logged as an error but not fatal. See {{ gh('wiki/model-loading-for-inference.md', 'model-loading-for-inference.md', anchor='conflict-detection') }}. |

## Cross-cutting issues and gaps

Pulling the broken cells out of the matrix gives this concrete TODO list:

1. **`_get_full_checkpoint_path` uses `os.path.isfile`** (both the explicit
   branch and the `best.ckpt` / `last.ckpt` fallback). This is the symmetric
   sibling of the `lightning_train.py` fix and the single biggest blocker:
   - `eval.checkpoint_path` / `eval.ckpt_path` = FSDP sharded dir → rejected.
   - `generation.ckpt_path` = FSDP sharded dir → rejected.
   - `hub_checkpoint_path` = FSDP sharded dir → rejected.
   - Eval auto-fallback never picks up `last.ckpt/` if it is a sharded dir.
   The training-side path already has `is_usable_lightning_train_checkpoint_path`
   in `xlm/utils/checkpoint_paths.py`; the inference-side check should be
   refactored on top of the same helpers.

2. **`extract_checkpoint` FSDP UX.** The `extract_checkpoint` command now dispatches on sharded dirs; [`extract-checkpoint.md`](extract-checkpoint.md) documents `apply_ema=false` and optional `max_shard_size`. Call {{ gh('src/xlm/utils/consolidate_model_checkpoint.py', 'consolidate_model_checkpoint') }} directly when you need a Harness-free export.

3. **No EMA on FSDP / sharded extract (by design).** `consolidate_model_checkpoint`
   only exports `state_dict` weights. For EMA-averaged publication, use a
   single-file full checkpoint and `extract_checkpoint` with `apply_ema=true`,
   or save EMA weights during training to a separate artifact.

4. **Hub upload paths differ.** `push_to_hub` / `Harness.push_to_hub` still
   serialize a **single** `model.safetensors` via `PyTorchModelHubMixin`.
   For multi-shard Hub uploads, consolidate locally with `max_shard_size` and
   upload the output folder via `HfApi.upload_folder`. Models above the
   single-file Hub limit need this path, not `job_type=push_to_hub` alone.

5. **Documentation drift.** Right now the relevant material is spread across
   four files and they don't agree:
   - [`docs/guide/extract-checkpoint.md`](extract-checkpoint.md) — covers both
     single-file and FSDP-sharded `extract_checkpoint`; see also
     {{ gh('src/xlm/utils/consolidate_model_checkpoint.py', 'consolidate_model_checkpoint') }} for Harness-free export.
   - [`docs/guide/push-to-hub.md`](push-to-hub.md) — single-file only;
     `model_only_checkpoint_path` is described but the sharded safetensors
     index variant is not.
   - {{ gh('wiki/model-loading-for-inference.md', 'wiki/model-loading-for-inference.md') }}
     — accurate for inference loading but does not enumerate which sources
     are accepted as **full** ckpts (and so does not flag the `os.path.isfile`
     gap).
   - [`docs/guide/llms.md` §5](llms.md#5-checkpointing-and-resuming) — FSDP
     sharded checkpoints, consolidation, and `extract_checkpoint` behavior.
   This page (`serialization.md`) is intended to subsume the cross-cutting
   parts; the others can shrink to feature-specific usage and link here.

## What "works as expected today" — short version

If you are on this page to figure out what is safe to rely on **right now**:

- **DDP / single-device, single-file `.ckpt`**: every workflow (train, resume,
  extract, eval, generate, push to hub) is supported.
- **FSDP sharded training and resume**: train, save, resume — supported,
  including auto-pickup of `last.ckpt/` and `on_exception.ckpt/` directories.
- **FSDP → model-only / Hub**: `xlm job_type=extract_checkpoint` (with
  `apply_ema=false`) on a sharded directory, or call
  {{ gh('src/xlm/utils/consolidate_model_checkpoint.py', 'consolidate_model_checkpoint') }}
  directly. Then use `model_only_checkpoint_path` pointing at `.safetensors` or
  `model.safetensors.index.json` for eval / generate / `push_to_hub` as needed.
- **HF Hub for inference (eval / generate)**: works for single-file,
  sharded safetensors, and legacy `pytorch_model.bin`, on both branches and
  revisions.

What is **not** safe to assume today, even though the configs and docs imply it:

- Pointing eval / generate / push_to_hub at an FSDP sharded directory
  (`*.ckpt/` with `*.distcp`) **silently fails** with "Checkpoint path does not exist".
- Asking `extract_checkpoint` to use **`apply_ema=true`** on an FSDP sharded
  directory **fails by design** — use a single-file checkpoint for EMA export.
- Publishing an **EMA-averaged** checkpoint that was saved **only** as FSDP
  sharded — use `state_dict_type: full` for a single-file export, or a separate
  EMA artifact; we do not merge EMA from sharded dirs in `extract_checkpoint`.
- Publishing a model **larger than the Hub single-file limit** via
  `job_type=push_to_hub` alone — `_save_pretrained` writes one file; consolidate
  with `max_shard_size` and upload the folder via `HfApi.upload_folder`.
