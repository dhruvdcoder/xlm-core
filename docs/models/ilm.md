# ILM — Insertion Language Model

## 1. Overview

`ilm` implements an insertion language model: training corrupts a sequence by dropping tokens at random positions, and the model is trained to predict the *multiset* of dropped tokens at each surviving position (sparse `target_ids` of shape `(B, L, V)`) and, optionally, the total dropped length via a classification head. At inference time the predictor inserts tokens at chosen positions until a stopping signal fires.

```bibtex
@misc{patel2025insertionlanguagemodelssequence,
  title = {Insertion Language Models: Sequence Generation with Arbitrary-Position Insertions},
  author = {Dhruvesh Patel and Aishwarya Sahoo and Avinash Amballa and Tahira Naseem and Tim G. J. Rudner and Andrew McCallum},
  year = {2025},
  eprint = {2505.05755},
  archivePrefix = {arXiv}
}
```

See [xlm-models/ilm/README.md](../../xlm-models/ilm/README.md).

## 2. Files at a glance

| Module | Public classes / helpers |
|---|---|
| [model_ilm.py](../../xlm-models/ilm/model_ilm.py) | `BaseRotaryTransformerILMModel`, `RotaryTransformerILMModel`, `RotaryTransformerITModel`, `RotaryTransformerILMModelWithClassification`, `RotaryTransformerILMModelWithStoppingClassification`, `RotaryTransformerILMModelWithLengthClassification`, GPT-2 variants (`BaseGPT2ILMModel`, `GPT2ILMModel`, `GPT2ILMModelWithClassification`, `GPT2ILMModelWithStoppingClassification`, `GPT2ILMModelWithLengthClassification`) |
| [loss_ilm.py](../../xlm-models/ilm/loss_ilm.py) | `ILMLossWithMaskedCE` |
| [predictor_ilm.py](../../xlm-models/ilm/predictor_ilm.py) | `ILMPredictorUtilitiesMixin`, `ILMPredictor`, `ILMPredictorWithLengthClassification`, `ILMPredictorWithStoppingClassification` |
| [datamodule_ilm.py](../../xlm-models/ilm/datamodule_ilm.py) | `DefaultILMCollator`, `ILMSeq2SeqCollator`, `ILMSeq2SeqPredCollator`, `ilm_drop_fn`, `ilm_single_segment_collate_target_fn`, `prepare_prefix_ids`, `prepare_target_ids_for_test`, `print_batch_ilm` |
| [nn.py](../../xlm-models/ilm/nn.py) | `remove_tokens`, `log_softmax_last_two_dims`, `masked_ce_last_two_dims`, `topk_over_last_two_dims`, `max_over_last_two_dims`, `sample_over_last_two_dims`, `general_sample_over_last_two_dims` |
| [metrics_ilm.py](../../xlm-models/ilm/metrics_ilm.py) | `mean_metric_update_fn`, `length_loss_metric_update_fn`, `token_ce_metric_update_fn` |
| [types_ilm.py](../../xlm-models/ilm/types_ilm.py) | `ILMBatch`, `ILMSeq2SeqPredictionBatch`, `ILMUncondtionalPredictionBatch`, `ILMInfillPredictionBatch`, `ILMLossDict`, `ILMModel` (Protocol), `ILMPredictionDict` |

## 3. Architecture

Two backbone families:

- **`BaseRotaryTransformerILMModel`** (`RotaryTransformerILMModel` etc.) — RoPE-based encoder. Concrete subclasses select what is returned:
  - `RotaryTransformerILMModel` -> `(vocab_logits, None)` (the base ILM).
  - `RotaryTransformerILMModelWithClassification` / `…WithStoppingClassification` / `…WithLengthClassification` -> `(vocab_logits, length_logits | classification_logits)`.
- **`BaseGPT2ILMModel`** + subclasses — GPT-2-style backbone (`xlm.modules.gpt2_transformer.GPT`) for the same set of head variants.

Common forward signature:

```python
forward(
    x_t: Integer[TT, " *batch seq_len"],
    attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
    positions: Optional[Integer[TT, " *batch seq_len"]] = None,
    token_type_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
    cls_position: Optional[Integer[TT, " *batch"]] = None,
) -> Tuple[
    Float[TT, " *batch seq_len vocab_size"],
    Optional[Float[TT, " *batch max_length | num_classes"]],
]
```

- `token_type_ids`: 0 for CLS, 1 for BOS/prefix, 2 for body tokens.
- `cls_position`: per-example CLS index used to pool the length-head representation.
- The base `RotaryTransformerILMModel` returns `(vocab_logits, None)`; the classification variants return a `length_logits` tensor pooled from the CLS position.

## 4. Batch contract

`ILMBatch` ([types_ilm.py](../../xlm-models/ilm/types_ilm.py)) — `post_seq_len` is the length **after** the random token drop:

| Field | Shape | Notes |
|---|---|---|
| `input_ids` | `(B, post_seq_len)` `int` | Surviving tokens after the drop. |
| `attention_mask` | `(B, post_seq_len)` `int` | 1 for real tokens. |
| `token_type_ids` | `(B, post_seq_len)` `int` | 0=CLS, 1=BOS/prefix, 2=body. |
| `target_ids` | `(B, post_seq_len, V)` `int` *or* `(B, target_seq_len)` `int` | Counts of dropped tokens at each surviving position (sparse). |
| `n_drops` | `(B, post_seq_len)` `bool` | True where a drop happened (equal to `target_ids.sum(dim=-1) > 0`). |
| `target_attention_mask` | `(B, target_seq_len)` `int` | Used by some seq2seq batches. |
| `cls_position` | `(B,)` `int` | CLS index (defaults to 0). |
| `constraint` | `(B, post_seq_len)` `bool` | Positions that should not be predicted (prediction only). |

## 5. Loss

`ILMLossWithMaskedCE(model, tokenizer, length_loss=None, length_loss_weight=None, stopping_class_weight=None, loss_on_padding=False, use_constraint=False, input_constraint=False)`:

- Constructor validation:
  - `stopping_class_weight` only valid when `length_loss="binary_ce"` -> else `ValueError`.
  - `loss_on_padding=True` raises `ValueError`.
  - `input_constraint=True` and `use_constraint=True` raise `NotImplementedError`.
- `configure(pl_module)` caches `mask_token_id_tensor`, validates `stopping_class_weight ∈ [0, 1]` and `length_loss_weight ∈ [0, 1]`, and converts both to tensors on the right device.
- The CE branch uses `masked_ce_last_two_dims` from [`ilm.nn`](../../xlm-models/ilm/nn.py): the model outputs `(B, post_seq_len, V)` logits and we compute CE against the sparse target counts at non-drop, non-pad positions.
- Optional length head:
  - `length_loss="ce"` -> standard CE on `length_logits`.
  - `length_loss="binary_ce"` -> per-class binary CE with `stopping_class_weight` weighting the two classes.
- `ILMLossDict`: `{loss, batch_loss, per_example_length_loss, per_example_ce, length_logits, n_drops}`.

## 6. Collators

The token-drop noising is implemented in `ilm_drop_fn` + `ilm_single_segment_collate_target_fn`; `_n_drop_uniformly` chooses the number of drops per example (sampled via the wired `NoiseSchedule`). All three collators below **require a real `NoiseSchedule`**.

| Class | Input | Output batch | Special behavior |
|---|---|---|---|
| `DefaultILMCollator` | `BaseCollatorInput` | `ILMBatch` | Pad-right to `block_size`, BOS/EOS optional, random token drops with `target_ids` as `(B, post_seq_len, V)` sparse counts. |
| `ILMSeq2SeqCollator` | `Seq2SeqCollatorInput` | `ILMBatch` (with `target_attention_mask`) | Prefix + suffix collation with token drops on the suffix only. |
| `ILMSeq2SeqPredCollator` | `Seq2SeqCollatorInput` | `ILMSeq2SeqPredictionBatch` | Prediction-time variant — `target_ids` carry the gold suffix; `input_ids` carry only the prefix. |

## 7. Predictor

Three classes in [predictor_ilm.py](../../xlm-models/ilm/predictor_ilm.py):

- **`ILMPredictor`** — base predictor, no length head. Iteratively selects an insertion position from the model's distribution over `(position, token)` pairs (using `topk_over_last_two_dims` / `sample_over_last_two_dims` from `ilm.nn`) and inserts one token per step.
- **`ILMPredictorWithLengthClassification`** — uses `length_logits` to decide when to stop (length head predicts remaining insertions).
- **`ILMPredictorWithStoppingClassification`** — uses a binary stopping head to decide stop per step.
- All three inherit utilities from `ILMPredictorUtilitiesMixin` (token sampling, decoding, history tracking via `PredictorHistoryMixin`).

Output `ILMPredictionDict`: `{text, text_with_spl_tokens, ids, attention_mask, positions, history, time_taken, loss=None}`.

## 8. Metrics

See [tests/models/ilm/test_metrics_ilm.py](../../tests/models/ilm/test_metrics_ilm.py).

| Function | Returned keys | Notes |
|---|---|---|
| `mean_metric_update_fn` | `value = loss_dict["batch_loss"].mean()` | Note this reads `batch_loss`, not `loss` (ILM convention). |
| `length_loss_metric_update_fn` | `value = loss_dict["per_example_length_loss"]` | Only meaningful when a length head is wired. |
| `token_ce_metric_update_fn` | `value = loss_dict["per_example_ce"]` | Token CE only (ignores length contribution). |

## 9. Configs / experiments

Hydra groups under [xlm-models/ilm/configs/](../../xlm-models/ilm/configs/). Available experiment entry points:

- `experiment=star_easy_ilm`, `experiment=star_medium_ilm`, `experiment=star_hard_ilm`
- `experiment=text_ilm`
- `experiment=lm1b_ilm`
- `experiment=owt_ilm` (recipe in the package README)

## 10. Testing

Tests in [tests/models/ilm/](../../tests/models/ilm):

- `test_model_ilm.py` — extends `BaseModelTests` and verifies that the base `RotaryTransformerILMModel` returns `(vocab_logits, None)`.
- `test_loss_ilm.py` — construction-time validation (`stopping_class_weight` requires `length_loss="binary_ce"`, `loss_on_padding=True` raises). A minimal-batch CE test is added by this plan once a sparse `ILMBatch` fixture is available.
- `test_collator_ilm.py` — construction smoke + `DefaultILMCollator(... noise_schedule=real_loglinear_schedule)` exercise (added in this plan).
- `test_predictor_ilm.py` — construction smoke (added in this plan).
- `test_metrics_ilm.py`, `test_nn_ilm.py` — pure-function helpers.

Shared fixtures (`tiny_ilm_model`, `ilm_batch`, `simple_tokenizer`, `real_loglinear_schedule`) live in [tests/conftest.py](../../tests/conftest.py) and [tests/models/ilm/conftest.py](../../tests/models/ilm/conftest.py).

## 11. API reference

- [`ilm.model_ilm`](../reference/ilm/model_ilm/)
- [`ilm.loss_ilm`](../reference/ilm/loss_ilm/)
- [`ilm.predictor_ilm`](../reference/ilm/predictor_ilm/)
- [`ilm.datamodule_ilm`](../reference/ilm/datamodule_ilm/)
- [`ilm.nn`](../reference/ilm/nn/)
- [`ilm.metrics_ilm`](../reference/ilm/metrics_ilm/)
- [`ilm.types_ilm`](../reference/ilm/types_ilm/)
