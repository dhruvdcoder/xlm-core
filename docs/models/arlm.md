# ARLM — Auto-Regressive Language Model

## 1. Overview

`arlm` is a standard left-to-right causal Transformer language model with rotary position embeddings: at each step the model predicts the next token given the prefix. The package supports both language-modeling and seq2seq tasks (prompt + target with a `-100`-ignored prompt region). See [xlm-models/arlm/README.md](../../xlm-models/arlm/README.md) for the high-level description.

## 2. Files at a glance

| Module | Public classes / helpers |
|---|---|
| [model_arlm.py](../../xlm-models/arlm/model_arlm.py) | `RotaryTransformerARLMModel` |
| [loss_arlm.py](../../xlm-models/arlm/loss_arlm.py) | `ARLMLoss` |
| [predictor_arlm.py](../../xlm-models/arlm/predictor_arlm.py) | `ARLMPredictor` |
| [datamodule_arlm.py](../../xlm-models/arlm/datamodule_arlm.py) | `DefaultARLMCollator`, `ARLMSeq2SeqCollator`, `ARLMSeq2SeqPredCollator`, `ARLMEmptyDataset`, `prepare_prefix_ids_arlm`, helpers |
| [metrics_arlm.py](../../xlm-models/arlm/metrics_arlm.py) | `seq2seq_exact_match_update_fn`, `seq2seq_token_accuracy_update_fn`, `mean_metric_update_fn`, `perplexity_metric_update_fn`, `token_nll_metric_update_fn`, `sequence_length_metric_update_fn`, `valid_tokens_metric_update_fn` |
| [types_arlm.py](../../xlm-models/arlm/types_arlm.py) | `ARLMBatch`, `ARLMSeq2SeqBatch`, `ARLMSeq2SeqPredictionBatch`, `ARLMLossDict`, `ARLMModel` (Protocol), `ARLMPredictionDict` |

## 3. Architecture

`RotaryTransformerARLMModel(num_embeddings, d_model, num_layers, nhead, ...)` is structurally similar to the MLM backbone (RoPE + `RotaryTransformerLayer`s + `RotaryTransformerFinalLayer`), but the loss path always supplies a **3-D causal attention mask**:

```python
forward(
    x_t: Integer[TT, " *batch seq_len"],
    attention_mask: Optional[Bool[TT, " *batch seq_len seq_len"]] = None,
    positions: Optional[Integer[TT, " *batch seq_len"]] = None,
    token_type_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
) -> Float[TT, " *batch seq_len vocab_size"]
```

- `attention_mask` accepts a `(B, L, L)` or `(B, L)` mask. The loss builds `causal_attention_mask = tril(ones(L, L)) & attention_mask.unsqueeze(1)` so future tokens are masked out.
- `positions` are computed as `attention_mask.cumsum(dim=1) - 1` (zeroed at padding) in the loss path.
- `token_type_ids` is accepted but currently passed straight through.

## 4. Batch contract

`ARLMBatch` ([types_arlm.py](../../xlm-models/arlm/types_arlm.py)):

| Field | Shape | Notes |
|---|---|---|
| `input_ids` | `(B, L)` `int` | Tokens (prompt + target for seq2seq). |
| `attention_mask` | `(B, L)` `int` | 1 for real tokens, 0 for padding. |
| `target_ids` | `(B, L)` `int` | Already **left-shifted by 1**; positions to ignore (prompt or padding) carry `-100`. |

Seq2seq variants (`ARLMSeq2SeqBatch`, `ARLMSeq2SeqPredictionBatch`) carry the same fields plus a `token_type_ids` channel for downstream code; the loss treats `-100` as the universal ignore signal.

## 5. Loss

`ARLMLoss(model, tokenizer)`:

- `configure(pl_module)` is a no-op (no `mask_token_id_tensor` cache needed).
- `loss_fn`:
  - Builds `positions = (attention_mask.cumsum(dim=1) - 1) * attention_mask`.
  - Builds `causal_attention_mask = attention_mask.unsqueeze(1) & tril(ones(L, L))` of shape `(B, L, L)`.
  - Runs `logits = model(input_ids, causal_attention_mask, positions)`.
  - Returns `loss = cross_entropy(logits.transpose(1, 2), target_ids, reduction="mean", ignore_index=-100)`.

Because `target_ids` already comes shifted by 1 (and prompt positions are `-100`), the loss reduces to next-token CE over the *target* segment.

## 6. Collators

`BaseCollatorInput` for LM training, `Seq2SeqCollatorInput` for seq2seq. The helper `prepare_prefix_ids_arlm` left-pads prompts and supports BOS/EOS placement.

| Class | Input | Output batch | Special behavior |
|---|---|---|---|
| `DefaultARLMCollator` | `BaseCollatorInput` | `ARLMBatch` | Pad-right to `block_size`, builds `target_ids = input_ids[1:] + [-100]` (last position and padding become `-100`). |
| `ARLMSeq2SeqCollator` | `Seq2SeqCollatorInput` | `ARLMSeq2SeqBatch` | Concatenates `[prompt][BOS][target][EOS]`; prompt positions in `target_ids` set to `-100`. |
| `ARLMSeq2SeqPredCollator` | `Seq2SeqCollatorInput` | `ARLMSeq2SeqPredictionBatch` | Left-padded prompt only as `input_ids`; target kept separate for evaluation. |

## 7. Predictor

`ARLMPredictor(max_steps, max_length, tokenizer, noise_schedule=None, sampling_method="sample", top=1000, p=0.9, model)`:

- `sampling_method`:
  - `"sample"` -> argmax/sample from full logits.
  - `"sample_top_k"` -> `sample_from_top_k(top, logits)`.
  - `"sample_top_p"` -> `sample_from_top_p(p, logits)`.
- `predict()` runs incremental greedy/sampled generation from the (left-padded) prompt, appending one token per step (or stopping early when EOS is produced or `max_length` reached).
- Output `ARLMPredictionDict`: `{text, text_with_spl_tokens, ids, attention_mask, positions, time_taken, output_start_idx}`.

The `noise_schedule` argument is accepted for interface symmetry with the other families and is unused.

## 8. Metrics

All `*_update_fn(batch, loss_dict, tokenizer=None)`. Worked examples: [tests/models/arlm/test_metrics_arlm.py](../../tests/models/arlm/test_metrics_arlm.py).

| Function | Returned keys | Notes |
|---|---|---|
| `seq2seq_exact_match_update_fn` | `pred`, `target`, `pred_length=None`, `target_length=None` | `pred = loss_dict["ids"][:, output_start_idx:]`. |
| `seq2seq_token_accuracy_update_fn` | `pred`, `target`, `pred_mask = ones_like(pred)` | All suffix tokens counted. |
| `mean_metric_update_fn` | `value = loss_dict["loss"]` | Generic scalar. |
| `perplexity_metric_update_fn` | `value = exp(nlls.mean())` | Uses `loss_dict["nlls"]`. |
| `token_nll_metric_update_fn` | `value = loss_dict["nlls"]` | Per-token NLL pass-through. |
| `sequence_length_metric_update_fn` | `value = attention_mask.sum(dim=1).float()` | Per-example sequence length. |
| `valid_tokens_metric_update_fn` | `value = (attention_mask & (target_ids != -100)).sum(dim=1).float()` | Excludes padding and ignored targets. |

## 9. Configs / experiments

Hydra groups under [xlm-models/arlm/configs/](../../xlm-models/arlm/configs/). Available experiment entry points:

- `experiment=star_easy_arlm`

## 10. Testing

Tests in [tests/models/arlm/](../../tests/models/arlm):

- `test_model_arlm.py` — extends `BaseModelTests`, plus a causal-mask leakage test (added in this plan).
- `test_loss_arlm.py` — extends `BaseLossTests`, plus a `-100` ignore-index test (added in this plan).
- `test_collator_arlm.py` — extends `BaseCollatorTests` and adds an ARLM-specific "target has ignore index" assertion.
- `test_predictor_arlm.py` — predictor smoke + vocab-range tests.
- `test_metrics_arlm.py` — pure-function helpers.

Shared fixtures (`tiny_arlm_model`, `arlm_batch`, `simple_tokenizer`) live in [tests/conftest.py](../../tests/conftest.py) and [tests/models/arlm/conftest.py](../../tests/models/arlm/conftest.py).

## 11. API reference

- [`arlm.model_arlm`](../reference/arlm/model_arlm/)
- [`arlm.loss_arlm`](../reference/arlm/loss_arlm/)
- [`arlm.predictor_arlm`](../reference/arlm/predictor_arlm/)
- [`arlm.datamodule_arlm`](../reference/arlm/datamodule_arlm/)
- [`arlm.metrics_arlm`](../reference/arlm/metrics_arlm/)
- [`arlm.types_arlm`](../reference/arlm/types_arlm/)
