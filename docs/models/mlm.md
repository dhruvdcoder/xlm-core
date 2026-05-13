# MLM — Masked Language Model

## 1. Overview

`mlm` implements a from-scratch rotary-Transformer masked language model in the BERT family: the model receives an input sequence with a fraction of positions replaced by `[MASK]` and is trained to recover the original tokens at those positions. The package ships standard padded-truncated training and prediction collators plus a packed-FlexAttention variant for protein and text training. See [xlm-models/mlm/README.md](../../xlm-models/mlm/README.md) for end-to-end recipes (UniRef50 standard / packed and OpenWebText packed).

## 2. Files at a glance

| Module | Public classes / helpers |
|---|---|
| [model_mlm.py](../../xlm-models/mlm/model_mlm.py) | `RotaryTransformerMLMModel` |
| [loss_mlm.py](../../xlm-models/mlm/loss_mlm.py) | `MLMLoss` |
| [predictor_mlm.py](../../xlm-models/mlm/predictor_mlm.py) | `MLMPredictor` |
| [datamodule_mlm.py](../../xlm-models/mlm/datamodule_mlm.py) | `DefaultMLMCollator`, `MLMSeq2SeqCollator`, `MLMSeq2SeqTrainCollator`, `MLMSeq2SeqPredCollator`, `_MLMSeq2SeqPredCollator`, `MLMInfillWithExactTargetPredCollator`, `DefaultInfillMLMCollator`, `PackedMLMCollator`, `MLMEmptyDataset`, `mlm_single_segment_collate_fn`, `prepare_prefix_ids`, `prepare_prefix_suffix_ids`, `print_batch_mlm` |
| [metrics_mlm.py](../../xlm-models/mlm/metrics_mlm.py) | `exact_match_update_fn`, `infill_token_accuracy_update_fn`, `seq2seq_exact_match_update_fn`, `seq2seq_token_accuracy_update_fn`, `mean_metric_update_fn` |
| [types_mlm.py](../../xlm-models/mlm/types_mlm.py) | `MLMBatch`, `PackedFlexMLMBatch`, `MLMSeq2SeqPredictionBatch`, `MLMLossDict`, `MLMModel` (Protocol), `MLMPredictionDict` |
| Family-private helpers | [history_mlm.py](../../xlm-models/mlm/history_mlm.py), [papl_unconditional.py](../../xlm-models/mlm/papl_unconditional.py), [unbatch.py](../../xlm-models/mlm/unbatch.py) |

## 3. Architecture

`RotaryTransformerMLMModel(num_embeddings, d_model, num_layers, nhead, ...)` is a stack of `RotaryTransformerLayer`s wrapped in a `RotaryTransformerLayerList` with a `RotaryEmbedding` cache, followed by `RotaryTransformerFinalLayer` projecting to the vocabulary.

```python
forward(
    x_t: Integer[TT, " *batch seq_len"],
    attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
    positions: Optional[Integer[TT, " *batch seq_len"]] = None,
    block_mask=None,
) -> Float[TT, " *batch seq_len vocab_size"]
```

- `attention_mask`: 1-D padding mask (True = valid token). Cast to `bool` internally.
- `positions`: per-token RoPE positions. For padded batches, the loss path computes them as `(attention_mask.cumsum(dim=1) - 1).clamp(min=0)`. For packed FlexAttention batches, positions are reset to 0 at each segment boundary.
- `block_mask`: a FlexAttention `BlockMask` produced for packed batches; when set, `attention_mask` is ignored. Toggled by `use_flex_attn=True`.

## 4. Batch contract

`MLMBatch` ([types_mlm.py](../../xlm-models/mlm/types_mlm.py)):

| Field | Shape | When present |
|---|---|---|
| `input_ids` | `(B, L)` `int` | always |
| `attention_mask` | `(B, L)` `bool` | padded batches; absent in packed FlexAttention |
| `target_ids` | `(B, L)` `int` | always — masks replaced by ground-truth tokens; `-100` at ignored positions when `loss_on_padding=False` |
| `positions` | `(B, L)` `int` | required for packed FlexAttention (RoPE reset per segment) |
| `segment_ids` | `(B, L)` `int` | packed batches only — feeds `mask_mod` for `BlockMask` |
| `block_mask` | `BlockMask` | packed batches when `model.use_flex_attn=True` |
| `fixed_positions_mask` | `(B, L)` `bool` | infill collators only — positions that must not be re-masked |

The packed FlexAttention variant uses `PackedFlexMLMBatch` (subset of the above) and `MLMLoss.__call__` builds the `BlockMask` on the training device.

## 5. Loss

`MLMLoss(loss_on_padding=False, loss_on_visible_tokens=False, model, tokenizer, use_num_masked_factor=False)`:

- `configure(pl_module)` caches `mask_token_id_tensor` on the right device.
- `__call__` builds a FlexAttention `BlockMask` from `segment_ids` (if `model.use_flex_attn=True` and the collator did not produce one), then delegates to `loss_fn`.
- `loss_fn` runs the model with the chosen attention path and computes:
  - `ignore = (input_ids != mask_token_id)` when `loss_on_visible_tokens=False` (default) — only masked positions count.
  - `ce = cross_entropy(logits_T, targets, reduction="none", ignore_index=-100)`.
  - Optional `1 / (num_masked + 1)` factor when `use_num_masked_factor=True` (uniform-per-example variance reduction).
  - Final loss = `masked_mean(ce.flatten(), ~ignore.flatten())`.
- Output: `MLMLossDict({"loss": scalar})`.

## 6. Collators

`BaseCollatorInput` = `{input_ids, attention_mask?, token_type_ids?}`; `Seq2SeqCollatorInput` = `{prompt_ids, input_ids, ...}`. The shared internal helper is `mlm_single_segment_collate_fn` (random per-example mask rate `t ~ U[0, 1]`).

| Class | Input | Output batch | Special behavior |
|---|---|---|---|
| `DefaultMLMCollator` | `BaseCollatorInput` | `MLMBatch` | Pad-right to `block_size`, BOS/EOS optional, random MLM masking. |
| `MLMSeq2SeqTrainCollator` | `Seq2SeqCollatorInput` | `MLMBatch` | Concatenates `[prompt][BOS][target][EOS]` with right padding; masks only suffix positions. |
| `MLMSeq2SeqCollator` | `Seq2SeqCollatorInput` | `MLMBatch` | Left-pads prompt and right-pads target separately (padding on both sides). |
| `_MLMSeq2SeqPredCollator` | `Seq2SeqCollatorInput` | `MLMBatch` | Same as `MLMSeq2SeqCollator` but masks **all** suffix tokens (`mask_all=True`); used for exact-match eval. |
| `MLMSeq2SeqPredCollator` | `Seq2SeqCollatorInput` | `MLMBatch` | `input_ids = left-padded prompt only`; `target_ids = right-padded target` (used for seq2seq prediction). |
| `MLMInfillWithExactTargetPredCollator` | `BaseCollatorInput` with pre-masked `prompt_ids` | `MLMBatch` | `mask_none=True` so existing masks in `prompt_ids` are kept; `target_ids` filled from `input_ids`. |
| `DefaultInfillMLMCollator` | `BaseCollatorInput` | `MLMBatch` | Like `DefaultMLMCollator` but restricts masking to positions where `prompt_ids[i] == mask_token_id`. |
| `PackedMLMCollator` | pre-packed `BaseCollatorInput` (EOS-separated) | `PackedFlexMLMBatch` | Builds `segment_ids`, per-segment `positions`, random MLM masking; **requires `use_flex_attn=True`**. |

## 7. Predictor

`MLMPredictor(max_steps, max_new_tokens=None, tokenizer, model, noise_schedule, top_k=None, top_p=None, confidence=None, threshold=None, skip_special_tokens=True)`:

- Sampling function is selected at `__init__`:
  - `top_k` only -> `sample_from_top_k`
  - `top_p` only -> `sample_from_top_p`
  - neither -> `sample_from_logits` (argmax-style)
  - both is rejected (`ValueError`)
- `predict()` clones `input_ids`, optionally appends `max_new_tokens` `[MASK]` tokens, derives positions from `attention_mask.cumsum-1`, then iterates `predict_single_step` until `stop()` returns true.
- `stop()` returns true when all examples have run out of `max_steps` or no `[MASK]` token remains.
- `predict_single_step(final_step=False)`:
  - When `confidence=None`: pick a uniform-random subset of masked positions of size `ceil(num_masked / steps_left)`.
  - When `confidence="prob_diff"`: select positions whose `top1 - top2` margin is smallest, threshold on cumulative low-confidence mass.
  - When `confidence="top_prob"`: same idea but on `1 - max(softmax)`.
  - `"entropy"` is declared but currently `NotImplementedError` inside the branch.
  - `final_step=True` unmasks every remaining `[MASK]`.
- Output `MLMPredictionDict`: `{text, ids, loss=None, time_taken, output_start_idx, steps_taken}`.

## 8. Metrics

`*_update_fn(batch, loss_dict, tokenizer=None)` callables fed to `MetricWrapper`. See [tests/models/mlm/test_metrics_mlm.py](../../tests/models/mlm/test_metrics_mlm.py) for worked examples.

| Function | Returned keys | Notes |
|---|---|---|
| `exact_match_update_fn` | `pred`, `target`, `pred_length=None`, `target_length=None` | Full-sequence comparison. |
| `infill_token_accuracy_update_fn` | `pred`, `target`, `pred_mask` | `pred_mask = (batch["input_ids"] == tokenizer.mask_token_id)`. |
| `seq2seq_exact_match_update_fn` | `pred = loss_dict["ids"][:, output_start_idx:]`, `target`, `pred_length`, `target_length` | Slices the generated suffix. |
| `seq2seq_token_accuracy_update_fn` | `pred`, `target`, `pred_mask = ones_like(pred)` | All suffix positions counted. |
| `mean_metric_update_fn` | `value = loss_dict["loss"]` | Generic scalar accumulator. |

## 9. Configs / experiments

Hydra groups under [xlm-models/mlm/configs/](../../xlm-models/mlm/configs/) (`collator/`, `datamodule/`, `experiment/`, `model/`, `model_type/`). Available experiment entry points:

- `experiment=star_easy_mlm`
- `experiment=sudoku_mlm`
- `experiment=sudoku_extreme_mlm`
- `experiment=lm1b_mlm`
- `experiment=owt_mlm`
- `experiment=owt_packed_mlm` (FlexAttention)
- `experiment=uniref50_packed_mlm` (FlexAttention, protein)

Recipes including packed-collator inspection (`debug=overfit`, `print_batch_fn=print_batch_mlm`) live in the package [README](../../xlm-models/mlm/README.md).

## 10. Testing

Tests live in [tests/models/mlm/](../../tests/models/mlm) and follow the 4-file mixin layout:

- `test_model_mlm.py` — extends `BaseModelTests`.
- `test_loss_mlm.py` — extends `BaseLossTests`.
- `test_collator_mlm.py` — extends `BaseCollatorTests`.
- `test_predictor_mlm.py` — predictor smoke + vocab-range tests, plus confidence-sampling coverage (added in this plan).
- `test_metrics_mlm.py`, `test_unbatch.py`, `test_papl_unconditional.py` — pure-function helpers.

Shared fixtures (`tiny_mlm_model`, `mlm_batch`, `simple_tokenizer`, `dummy_noise_schedule`) live in [tests/conftest.py](../../tests/conftest.py) and [tests/models/conftest.py](../../tests/models/conftest.py).

## 11. API reference

- [`mlm.model_mlm`](../reference/mlm/model_mlm/)
- [`mlm.loss_mlm`](../reference/mlm/loss_mlm/)
- [`mlm.predictor_mlm`](../reference/mlm/predictor_mlm/)
- [`mlm.datamodule_mlm`](../reference/mlm/datamodule_mlm/)
- [`mlm.metrics_mlm`](../reference/mlm/metrics_mlm/)
- [`mlm.types_mlm`](../reference/mlm/types_mlm/)
