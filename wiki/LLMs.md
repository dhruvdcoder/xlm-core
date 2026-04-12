# PR: LLM Evaluation Support for xlm-core

**Branch:** `dhruvesh/LLMs`

This PR extends xlm-core to support evaluation of large language models (LLMs), primarily targeting decoder-only models like Dream. The changes make the training harness flexible enough to handle eval-only runs (no loss function, no training metrics) and add concrete support for the MATH-500 benchmark as a first downstream task.

---

## Summary of Changes (`src/xlm/`)

### 1. Eval-Only Harness Support (`harness.py`)

The `Harness` class previously assumed a loss function, diagnostic metrics, and reported metrics were always present. This is true for training but breaks eval-only workflows where we only need prediction + post-hoc scoring.

**Changes:**

- **Optional loss function:** `loss_function` is now `Optional[LossFunction]`. If `loss` is missing from the config, the harness skips loss instantiation. A guard on `compute_loss` raises a clear error if an LM-loss dataloader is used without a configured loss.
- **Optional metrics:** `diagnostic_metrics` and `reported_metrics` configs now default to `null`. `instantiate_metrics` uses `OmegaConf.select` with a `default=None` fallback so missing keys don't crash Hydra instantiation.
- **Compile guard:** `torch.compile` setup is skipped when `loss_function is None`, since there is nothing to compile in eval-only mode.
- **Post-hoc evaluation output:** `compute_post_hoc_metrics` no longer mutates the original predictions JSONL. Instead, results (per-sample scores + aggregated metrics) are written to a separate `results_epoch=…_step=….json` file. The `update_logged_predictions` parameter has been removed.
- **Dataloader name forwarded:** Post-hoc evaluators now receive `dataloader_name` as a kwarg, enabling routing in composite evaluators.

### 2. `EvalDatasetManager` (`datamodule.py`)

A new lightweight `EvalDatasetManager` class for eval-only dataset splits.

**Motivation:** The existing `DatasetManager` carries a lot of machinery for training (iterable datasets, manual disk caching, on-the-fly processors, DDP sharding). Eval datasets are small, map-style, and need none of that. The manual cache is particularly problematic for eval because cache paths don't distinguish tokenizer/model, leading to stale cache bugs when switching models.

**Key properties:**
- Map-style datasets only (no iterable dataset support).
- No manual disk cache; relies on HF `Dataset.map` auto-cache (controlled via `load_from_cache_file`).
- No on-the-fly processors — all preprocessing happens in `prepare_data` via `preprocess_function`.
- Implements the same duck-typed API as `DatasetManager` (`prepare_data`, `setup`, `get_dataloader`, `set_epoch`) so `TextDataModule` can use it transparently.
- `stages` parameter controls which Lightning stages this dataset participates in (defaults to `["validate", "test"]`).

**Config:** A `default_eval.yaml` Hydra config is provided as a base for eval dataset managers.

**Type updates in `TextDataModule`:** The `dataset_managers` dict now accepts `Union[DatasetManager, EvalDatasetManager, UnconditionalGenerationDatasetManager]`, and all iteration sites (`prepare_data`, `setup`, `set_epoch`) have updated type annotations.

### 3. MATH-500 Evaluation Task (`tasks/math500.py`)

A new evaluation task module for the [MATH-500 benchmark](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).

**`Math500Eval` (post-hoc evaluator):**
- Reads `text` (model generation) and `truth`/`answer` (gold answer) from logged predictions.
- Uses `math_verify.parse` to extract structured mathematical expressions and `math_verify.verify` to check equivalence.
- Reports `math_equal_at_1` accuracy, parse failure counts, verify failure counts, and optional timing/step statistics.

**`math500_preprocess_fn` (dataset preprocessor):**
- Builds fewshot prompts matching the lm-evaluation-harness format (`Problem: … Answer: …`).
- Tokenizes the prompt into `prompt_ids` / `target_ids` for the prediction pipeline.
- Carries the gold `answer` field through to predictions via `additional_fields_from_batch`.
- Fewshot examples are loaded from the dataset itself (configurable split/seed) and cached.

**Configs:**
- `datasets/math500.yaml` — `DatasetManager`-based config for MATH-500.
- `datasets/math500_test.yaml` — `EvalDatasetManager`-based config (lighter-weight, recommended).

### 4. Composite Post-Hoc Evaluator (`tasks/composite_eval.py`)

A new `CompositePostHocEvaluator` that routes `eval()` calls to task-specific evaluators based on dataloader name.

**Motivation:** When running eval on multiple tasks simultaneously (e.g. MATH-500 + molecule generation), each prediction dataloader needs different evaluation logic. The composite evaluator maps dataloader name patterns to evaluator instances.

**Behavior:**
- Takes a dict of `{pattern: evaluator}` pairs.
- On `eval()`, matches the `dataloader_name` against patterns (substring match) and delegates to the first matching evaluator.
- Falls back to the single evaluator if only one is registered, or returns empty metrics with a warning if no match is found.

### 5. Sharded Safetensors Support (`utils/hf_hub.py`)

Extended the HF Hub weight loading pipeline to handle sharded safetensors checkpoints (common for large LLMs like Qwen-7B, LLaMA, etc.).

**New download path:** `download_model_weights` now tries: (1) single `model.safetensors`, (2) sharded safetensors (`model.safetensors.index.json` + shard files), (3) `pytorch_model.bin`.

**Shard-by-shard loading:** `_load_sharded_safetensors_into_model` loads one shard at a time into the model, keeping peak RAM at approximately model size + one shard (instead of model size + full checkpoint). A strict-mode check is performed at the end by comparing checkpoint vs model keys.

**Full state dict fallback:** `_state_dict_from_safetensors_index` merges all shards into a single state dict for the legacy `load_model_state_dict_from_file` path (higher peak RAM, but needed for existing callers).

### 6. `skip_init_weights` & `init_dtype` for Large Model Loading (`commands/lightning_train.py`, `utils/model_loading.py`)

**`skip_init_weights`:** When loading pretrained weights, random initialization of the model is wasteful (and slow for large models). Setting `skip_init_weights: true` in the config wraps model instantiation in `transformers.modeling_utils.no_init_weights()`, skipping random init entirely. Only activates when a checkpoint path is provided.

**`init_dtype`:** A new config key that sets `torch.set_default_dtype` during model construction, allowing models to be instantiated directly in `float16` or `bfloat16` instead of `float32` (important for fitting large models in GPU memory).

Both are supported in both the training entrypoint (`lightning_train.py`) and the inference loader (`model_loading.py`).

### 7. Sampling Temperature Support (`utils/nn.py`)

`sample_from_top_k` and `sample_from_top_p` now accept an optional `temperature` parameter (default 1.0). Temperature scaling is applied before top-k/top-p filtering, following standard practice.

### 8. Generalized Trailing Token Removal (`utils/text.py`)

`remove_trailing_pads` is generalized to strip arbitrary trailing special tokens, not just `pad_token`. It now accepts an optional `tokens_to_remove` list and matches longest-suffix-first to handle cases like `" </s>"` vs `"</s>"`.

### 9. `Seq2SeqCollatorInput` Update & `additional_fields_from_batch` Fix (`datamodule.py`, `log_predictions.py`)

- **`Seq2SeqCollatorInput`:** Refactored to use `NotRequired` (PEP 655) for `input_ids` and `target_ids`, supporting prompt-only batches (where the model generates the full response) alongside traditional prompt+target batches. Updated docstring clarifies the role of each field.
- **`additional_fields_from_batch`:** When `LogPredictions` is configured with `additional_fields_from_batch` (e.g. to carry `answer` through for post-hoc eval), those fields are now automatically appended to each writer's `fields_to_keep_in_output` list so they aren't dropped during output serialization.

### 10. DatasetManager Cache Control (`datamodule.py`)

Added `preprocess_load_from_cache_file` and `on_the_fly_load_from_cache_file` parameters to `DatasetManager`, passed through to `Dataset.map(load_from_cache_file=…)`. This allows disabling HF auto-cache when the preprocessing function has changed but the hash hasn't (common with tokenizer-dependent transforms).

### 11. Miscellaneous

- **`DeNovoEval` / `FragmentEval`:** Added `**kwargs` to `eval()` signatures for forward-compatibility with `CompositePostHocEvaluator` (which passes `dataloader_name`).
- **Config defaults:** `diagnostic_metrics: null` and `reported_metrics: null` added to `config.yaml` so they don't need to be explicitly set in eval-only experiments.
- **`default_hf.yaml` callback config:** New callback defaults preset (excludes HF-specific callbacks).
- **`requirements.txt`:** Added `math-verify` dependency.
- **`requirements/llm_eval.txt`:** New requirements file for LLM evaluation with the ANTLR4 variant of math-verify.
- **Formatting:** Various PEP 8 / Black formatting fixes throughout.

---

## Summary of Changes (`xlm-models/`)

### 12. Dream Model Architecture (`dream/`)

The Dream model type consists of two layers on top of the upstream `DreamModelCore` backbone (from `xlm.backbones.dream`):

**`DreamModel`:** Inherits from `DreamModelCore`, sets `config_class = DreamConfig`, and serves as the pure forward-pass model (`input_ids -> MaskedLMOutput`). No generation logic lives in the model itself — all generation is handled by the predictor.

**`DreamXLMModel`:** Subclass of `DreamModel` that adapts the forward signature to the xlm-core MLM predictor protocol (`x_t, attention_mask, positions -> logits`). It converts a 2D `attention_mask` to the 4D format Dream's attention layers expect and returns raw logits (not `MaskedLMOutput`). HF checkpoints (e.g. Dream-v0-Instruct-7B) load directly without key-prefix issues since the weight names are identical to `DreamModelCore`.

### 13. Dream Predictor (`dream/predictor_dream.py`)

`DreamPredictor` is a fully-featured diffusion predictor (~600 lines) built on the same architecture as `MLMPredictor`. It implements the xlm-core `Predictor[MLMBatch, MLMPredictionDict]` protocol so that all harness features (logging, post-hoc evaluation, callbacks) work out of the box.

**Key features:**

- **Token sampling:** Supports `temperature`, `top_k`, `top_p`, and greedy (`temperature=0`) sampling, using the same `sample_from_top_k`/`sample_from_top_p`/`sample_from_logits` utilities as MLM.
- **Confidence-based unmasking:** Supports `top_prob`, `prob_diff`, and `entropy` confidence metrics for position selection, with configurable `confidence_temperature` for stochastic vs greedy position ordering.
- **`LogitsShiftBy1`:** A callable logits hook that shifts next-token logits to align with per-position predictions (Dream-style `logits[i] = backbone_output[i+1]`). Defined in `mlm.predictor_mlm` and re-exported from `dream/__init__.py`.
- **`BlockAllowSchedule`:** Unmasking schedule that reveals one block of positions at a time (e.g. 16 tokens per block), preventing the model from attending to not-yet-ready future blocks. Requires `max_new_tokens` to be a multiple of `block_size` and `block_size` to divide evenly into diffusion steps.
- **`generate_until` / early stopping:** Accepts a list of stop strings (e.g. `<|endoftext|>`, `"Problem:"`). At each step, tokenized patterns are checked against the current output using an efficient batched `unfold`-based match. Generation stops when any pattern appears in the output with no mask tokens before it. A `force_fill_max` counter ensures termination even if masks remain (defaults to 3 hits).
- **`max_new_tokens`:** Appends mask tokens to the prompt, enabling open-ended generation where the suffix length is not predetermined.
- **Debug printing:** When `flags.DEBUG_PRINT_PREDS` is set, prints intermediate and final decoded outputs per step.
- **Per-sample timing and step counts:** Returns `time_taken` and `steps_taken` per sample in the prediction dict.

### 14. MLM Predictor Enhancements (`mlm/predictor_mlm.py`)

Changes to the MLM predictor that mirror the Dream predictor's new features. These are backward-compatible — all new parameters have defaults that preserve existing behavior.

**New parameters (all optional with backward-compatible defaults):**

- `temperature` (default `1.0`): Token sampling temperature. `temperature=0` enables greedy decoding via `top_k=1`.
- `confidence_temperature` (default `1.0`): Controls stochastic vs greedy position selection when using confidence-based unmasking. Previously only threshold-based selection was supported.
- `logits_hook` (default `None`): Arbitrary transform on model logits before sampling. Enables Dream-style shifted logits on MLM models without code changes.

**Refactored internals:**

- **`_require_tokenizer()`:** Lazy tokenizer validation (replaces eager check in `__init__`), allowing `tokenizer=None` at construction time (Harness sets it later).
- **`_compute_confidence()`:** Factored out confidence score computation (shared with Dream predictor).
- **Confidence-based unmasking:** Two code paths now exist: (1) legacy threshold-based (`confidence + threshold`) for backward compatibility, and (2) new score-based (`confidence` without `threshold`) using `select_random_indices` with confidence scores and temperature. The `threshold is None` check was previously an error; it now routes to the new path.
- **`attention_mask` handling:** Uses `.get("attention_mask")` with a fallback to all-ones, instead of assuming the key exists. Casts to `bool` dtype explicitly.
- **Debug printing:** Same `flags.DEBUG_PRINT_PREDS` support as Dream predictor.

**`LogitsShiftBy1`:** New class (also in `mlm/predictor_mlm.py`) — see description in Dream section above. Exported from `mlm/__init__.py`.

> **Potential breaking change (MLM):** The `threshold` parameter is no longer required when `confidence` is set. Old configs that set `confidence` without `threshold` previously raised `ValueError`; they now silently use the new score-based path. This is the intended behavior but may produce different results if a config was relying on the error to catch misconfigurations.

### 15. Seq2Seq Collator Flexibility (`mlm/datamodule_mlm.py`)

The seq2seq collators (`MLMSeq2SeqTrainCollator`, `MLMSeq2SeqCollator`, `MLMSeq2SeqPredCollator`, `_MLMSeq2SeqPredCollator`) previously hard-coded the field names `prompt_ids` and `input_ids`. They now support configurable field names and pass-through fields.

**New collator parameters (all with backward-compatible defaults):**

- `prompt_field` (default `"prompt_ids"`): Key for the prompt token ids in each example.
- `target_field` (default `"target_ids"`): Key for the suffix/target token ids. Falls back to `input_ids` when `target_field="target_ids"` and the row has no `target_ids` key — this preserves backward compatibility with existing datasets that use `input_ids` as the suffix field.
- `pass_through_fields` (default `["answer", "target"]`): List of keys to copy from the raw examples into the batch dict unmodified. This is how gold answers (e.g. MATH-500 `answer`) flow from the dataset through collation into the batch so `LogPredictions.additional_fields_from_batch` can pick them up.

**`seq2seq_suffix_ids()` helper:** Centralized suffix extraction with the `target_ids` -> `input_ids` fallback logic.

**Prompt-only batches in `MLMSeq2SeqPredCollator`:** When every example in the batch has an empty suffix (e.g. open-ended generation with no target), the collator now returns a batch without `target_ids` instead of crashing. Mixed batches (some empty, some non-empty suffixes) raise `ValueError`.

**`print_batch_mlm`:** Now guards `target_ids` access with `if "target_ids" in batch`, avoiding `KeyError` on prompt-only batches.

**Batch construction:** Collators now build batch dicts via plain `dict` literals + `_merge_pass_through()` instead of calling `MLMBatch(...)` directly. This is needed because pass-through fields (like `answer`) are not part of the `MLMBatch` TypedDict.

> **Potential breaking change (MLM):** Existing configs with custom collators that relied on `input_ids` as the suffix field still work (the `target_ids` -> `input_ids` fallback handles it). However, the batch dict shape changes subtly: `MLMSeq2SeqPredCollator` may now omit `target_ids` entirely for prompt-only batches. Code that indexes `batch["target_ids"]` without checking existence will break on these batches.

### 16. `MLMBatch` / `MLMSeq2SeqPredictionBatch` Type Updates (`mlm/types_mlm.py`)

- `MLMSeq2SeqPredictionBatch.target_ids` is now `NotRequired` (PEP 655), reflecting that prediction batches may be prompt-only (no suffix to predict against).
- Added `NotRequired` import with `typing_extensions` fallback for Python < 3.11.

### 17. Dream Model Registration & Configs

**`xlm_models.json`:** Added `"dream": "dream"` entry so the Dream model type is discoverable by the xlm-core plugin system.

**New Hydra configs:**

- `model_type/dream_base.yaml` — Shared base: sets `Harness` as the lightning module, `DreamPredictor` as the predictor with `LogitsShiftBy1` logits hook.
- `model_type/dream.yaml` — Training variant: composes `dream_base` + `accumulated_loss` reported metrics for train/val/test. Existing config, now simplified to a 6-line file composing `dream_base`.
- `model_type/dream_eval.yaml` — Eval-only variant: composes `dream_base` with no additional metrics.
- `model/dream_7b.yaml` — Model config for `DreamXLMModel` matching the [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) architecture (Qwen2-based, 28 layers, GQA, RoPE).
- `collator/math500_pred_dream.yaml` — `MLMSeq2SeqPredCollator` configured for MATH-500 with `pass_through_fields: [answer, target]`.
- `datamodule/math500_dream.yaml` — Datamodule composing the MATH-500 eval dataset with the Dream collator for val and test splits.
- `experiment/math500_dream_eval.yaml` — Full experiment config for MATH-500 eval on Dream-7B: loads the Dream tokenizer, sets `init_dtype: bfloat16`, `skip_init_weights: true`, 512 diffusion steps, block-based unmasking (block_size=16), top_prob confidence, generate_until stop tokens, and `Math500Eval` post-hoc evaluator.
- `debug/math500_debug.yaml` — Debug variant with reduced parameters for local testing.

**`dream/__init__.py`:** Exports `DreamConfig`, `DreamModel`, `DreamXLMModel`, `DreamPredictor`, `LogitsShiftBy1`, and `print_batch_dream`.

**`dream/datamodule_dream.py`:** New file with `print_batch_dream` — a debug batch printer that logs the first example's decoded prompt for Dream/MLM prediction batches.

### 18. Backward Compatibility Summary (MLM)

All MLM changes are designed to be backward-compatible. The table below summarizes the risks:

| Change                                                              | Risk                                                          | Mitigation                                                     |
|---------------------------------------------------------------------|---------------------------------------------------------------|----------------------------------------------------------------|
| `confidence` without `threshold` no longer errors                   | Low — new code path activates instead of crash                | Old configs with both `confidence` + `threshold` are unchanged |
| `MLMSeq2SeqPredCollator` may omit `target_ids`                      | Medium — only affects prompt-only batches                     | Only triggers when all suffixes are empty (new use case)       |
| Batch dicts built via `dict()` not `MLMBatch()`                     | Low — structurally identical                                  | Pass-through fields are extra keys; existing code ignores them |
| `tokenizer=None` allowed at init                                    | None — Harness always sets tokenizer before use               | `_require_tokenizer()` raises if used before set               |
| `target_field` defaults to `"target_ids"` with `input_ids` fallback | None — existing data uses `input_ids` and fallback handles it | Explicit `target_field="input_ids"` can be set if needed       |
