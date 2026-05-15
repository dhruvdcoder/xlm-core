---
name: MAUVE post-hoc xlm-core
overview: Add a new post-hoc evaluator to xlm-core that mirrors proseco’s MAUVE (and optionally token-entropy) scoring over logged prediction JSONL, following the same `eval(predictions, tokenizer, ...)` contract as `Math500Eval`, with an optional setuptools extra for the MAUVE dependency.
todos:
  - id: impl-mauve-eval
    content: Add MauveTextEval (or equivalent) in src/xlm/tasks/ with eval() contract, lazy mauve import, p_text vs p_features paths, optional entropy
    status: pending
  - id: packaging-extra
    content: Add requirements/mauve_extra.txt + setup.py extras_require and all-union; document pip install
    status: pending
  - id: hydra-yaml
    content: Add src/xlm/configs/lightning_train/post_hoc_evaluator/mauve_text.yaml and optional example experiment defaults
    status: pending
  - id: tests
    content: Add pytest with mocked mauve.compute_mauve (+ optional entropy unit test)
    status: pending
  - id: docs
    content: Document fields, LogPredictions.additional_fields_from_batch, pairing with generative_perplexity in metrics or wiki eval
    status: pending
isProject: false
---

# MAUVE / text-quality post-hoc evaluator for xlm-core

## Context

- **Target behavior** comes from proseco’s `_gen_eval` rank-0 block in [`proseco/main.py`](proseco/main.py): MAUVE via `mauve.compute_mauve` with either precomputed `p_features` (`.npy`) or human reference strings `p_text`, plus optional **mean token-distribution entropy** over generations using the run’s `tokenizer` and `torch.special.entr`. Generative PPL there uses a separate LM and multi-GPU aggregation; **do not duplicate that inside `post_hoc_evaluator`**—xlm-core already has [`Harness.compute_generative_perplexity`](xlm-core/src/xlm/harness.py) + [`GenerativePerplexityEvaluator`](xlm-core/src/xlm/generative_perplexity.py); document pairing them in YAML instead.

- **Integration point**: [`Harness.compute_post_hoc_metrics`](xlm-core/src/xlm/harness.py) loads JSONL, calls `self.post_hoc_evaluator.eval(predictions, tokenizer=self.tokenizer, dataloader_name=dataloader_name)`, logs `aggregated_metrics`, and writes `results_epoch=…_step=….json`. Your evaluator must return `Tuple[List[Dict], Dict[str, Any]]` like [`Math500Eval.eval`](xlm-core/src/xlm/tasks/math500.py).

```mermaid
flowchart LR
  jsonl[Logged predictions JSONL]
  harness[Harness.compute_post_hoc_metrics]
  eval[MauveTextEval.eval]
  out[results JSON plus Lightning logs]
  jsonl --> harness --> eval --> out
```

## Recommended design

### 1. New evaluator class

- **Module**: e.g. [`xlm-core/src/xlm/tasks/mauve_text_eval.py`](xlm-core/src/xlm/tasks/mauve_text_eval.py) (name bikesheddable) with a single class e.g. `MauveTextEval`.

- **Constructor (Hydra-friendly)**:
  - `reference_field: str` — batch key copied into each prediction (via `LogPredictions.additional_fields_from_batch`) holding the human/reference string for `p_text`. **Default proseco behavior that reloads the validation dataloader inside the evaluator should be avoided** (tight coupling, rank/dataloader ambiguity); document this as the supported path.
  - `generation_field: str = "text"` — model output for `q_text`.
  - `p_features_path: Optional[str] = None` — if set, `numpy.load` and assert row count matches `len(predictions)` (same semantics as proseco lines 391–394).
  - `max_text_length: int = 1024`, `device_id: int = 0`, `featurize_model_name: Optional[str] = None` (pass through to MAUVE if the API supports it), `verbose: bool = False`.
  - `compute_entropy: bool = False` — when true and `tokenizer` is not `None`, for each prediction append a per-row field (e.g. `token_entropy`) and add `mean_token_entropy` to `aggregated_metrics` using the same formula as proseco (lines 376–384).

- **`eval` implementation**:
  - Lazy-import `mauve` (and `numpy` if needed) inside `eval`; if import fails, raise a clear `ImportError` pointing at the new extra (same spirit as `math_verify` in [`math500.py`](xlm-core/src/xlm/tasks/math500.py)).
  - Build `q_text` from `generation_field`. Build `p_text` from `reference_field` **or** use `p_features`; validate mutual exclusion / required fields with explicit errors.
  - Call `mauve.compute_mauve(...)`, then set `aggregated_metrics` with at least `mauve`, `num_samples`; optionally include `num_buckets` or other scalars from the result object if useful and JSON-serializable (avoid dumping huge arrays unless behind a flag).
  - MAUVE is a **global** score; per-sample dicts need not get a MAUVE field unless you want a repeated scalar for debugging.

### 2. Packaging

- Add [`xlm-core/requirements/mauve_extra.txt`](xlm-core/requirements/mauve_extra.txt) (or similar) listing the same MAUVE package proseco uses (confirm name on PyPI: typically `mauve-text` or project-specific—**verify against proseco’s environment** before pinning).
- Extend `extras_require` in [`xlm-core/setup.py`](xlm-core/setup.py) with e.g. `"mauve": load_requirements_optional("requirements/mauve_extra.txt")` and include it in the `"all"` union so CI can optionally install it.
- Update the long_description install blurb in `setup.py` to mention `pip install "xlm-core[mauve]"`.

### 3. Hydra config snippet

- Add [`xlm-core/src/xlm/configs/lightning_train/post_hoc_evaluator/mauve_text.yaml`](xlm-core/src/xlm/configs/lightning_train/post_hoc_evaluator/mauve_text.yaml) mirroring [`denovo.yaml`](xlm-core/src/xlm/configs/lightning_train/post_hoc_evaluator/denovo.yaml): `_target_`, sensible defaults, `reference_field` placeholder.

- For **multi-task** setups, document wiring under [`CompositePostHocEvaluator`](xlm-core/src/xlm/tasks/composite_eval.py) with a dataloader-name substring key (same pattern as [`math500.py` docstring](xlm-core/src/xlm/tasks/math500.py)).

### 4. Experiment YAML expectations

- Set `log_predictions.additional_fields_from_batch` to include the chosen `reference_field` so each JSONL row carries human text aligned with `text` (same mechanism as MATH-500’s `answer` field in [`math500_dream_eval.yaml`](xlm-core/xlm-models/dream/configs/experiment/math500_dream_eval.yaml)).

### 5. Tests and docs

- **Tests**: New test under `xlm-core/tests/` that **mocks** `mauve.compute_mauve` (no GPU, no HF download) and asserts correct arguments and aggregate keys. Optionally a second test for entropy with a tiny fake tokenizer.

- **Docs**: Short addition to [`xlm-core/docs/guide/metrics.md`](xlm-core/docs/guide/metrics.md) or [`xlm-core/wiki/eval.md`](xlm-core/wiki/eval.md) describing required prediction fields, optional `p_features_path` ordering constraint, and dependency extra.

## Risks / constraints

- **GPU and downloads**: MAUVE featurizers may require CUDA and model weights; document `device_id` and offline/cache expectations.
- **Results JSON size**: `compute_post_hoc_metrics` writes full `predictions` into the results file; large `text` + many fields increases disk use.
- **Distributed training**: Post-hoc metrics are logged `rank_zero_only=True` in the harness; avoid duplicate heavy work by keeping MAUVE computation inside `eval` only when the harness invokes it on rank 0 (current pattern for other evaluators).

## Out of scope (explicit)

- Porting proseco’s **distributed generative PPL** loop into this evaluator (use existing `generative_perplexity` config instead).
- NFE aggregation from proseco (training-harness concern, not MAUVE post-hoc).
