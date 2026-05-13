# MDLM — Masked Diffusion Language Model

## 1. Overview

`mdlm` implements a continuous-time absorbing-state discrete diffusion language model. Training samples a per-example time \(t \in [\epsilon, 1]\), maps it through a noise schedule \(\sigma(t)\) to a corruption rate, masks tokens with that rate, and trains the model to denoise with a KL-weighted cross-entropy. The backbone is a DDiT-style Transformer with AdaLN time conditioning and rotary positional embeddings.

```bibtex
@misc{sahoo2024simpleeffectivemaskeddiffusion,
  title = {Simple and Effective Masked Diffusion Language Models},
  author = {Subham Sekhar Sahoo and Marianne Arriola and Yair Schiff and Aaron Gokaslan and Edgar Marroquin and Justin T Chiu and Alexander Rush and Volodymyr Kuleshov},
  year = {2024},
  eprint = {2406.07524},
  archivePrefix = {arXiv}
}
```

See [xlm-models/mdlm/README.md](../../xlm-models/mdlm/README.md).

## 2. Files at a glance

| Module | Public classes / helpers |
|---|---|
| [model_mdlm.py](../../xlm-models/mdlm/model_mdlm.py) | `BaseMDLMModel`, `MDLMModel` |
| [loss_mdlm.py](../../xlm-models/mdlm/loss_mdlm.py) | `MDLMLoss` |
| [predictor_mdlm.py](../../xlm-models/mdlm/predictor_mdlm.py) | `MDLMPredictor` |
| [datamodule_mdlm.py](../../xlm-models/mdlm/datamodule_mdlm.py) | `DefaultMDLMCollator`, `MDLMSeq2SeqTrainCollator`, `MDLMSeq2SeqPredCollator`, `MDLMEmptyDataset`, `mdlm_single_segment_collate_fn` |
| [noise_mdlm.py](../../xlm-models/mdlm/noise_mdlm.py) | `ContinousTimeNoiseSchedule`, `ContinuousTimeLinearSchedule`, `ContinuousTimeLogLinearSchedule`, `_convert_to_correlated` |
| [metrics_mdlm.py](../../xlm-models/mdlm/metrics_mdlm.py) | `seq2seq_exact_match_update_fn`, `seq2seq_token_accuracy_update_fn`, `mean_metric_update_fn` |
| [types_mdlm.py](../../xlm-models/mdlm/types_mdlm.py) | `MDLMBatch`, `MDLMSeq2SeqPredictionBatch`, `MDLMLossDict`, `MDLMModel` (Protocol), `MDLMPredictionDict` |

## 3. Architecture

`MDLMModel(num_embeddings, d_model, num_layers, nhead, ...)` wraps a `DDiTLayerList` (`DDiTLayer` blocks with AdaLN time conditioning) around a `TimestepEmbedder` and projects through `DDitFinalLayer`. The forward signature differs from MLM in that it takes a per-example total-noise value, encoded as the AdaLN condition vector:

```python
forward(
    x_t: Integer[TT, " *batch seq_len"],
    noise: Float[TT, " *batch"],            # ``total_noise`` (passed positionally)
    attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
    positions: Optional[Integer[TT, " *batch seq_len"]] = None,
) -> Float[TT, " *batch seq_len vocab_size"]
```

- `noise` is the conditioning signal; internally `c = SiLU(TimestepEmbedder(noise))`.
- `positions` are derived from `attention_mask.cumsum(dim=1) - 1` when `None`.
- `attention_mask` is cast to `bool` internally.

The Protocol [`MDLMModel`](../../xlm-models/mdlm/types_mdlm.py) in `types_mdlm.py` uses `total_noise: Float[TT, " batch"]` as the second argument; `MDLMLoss` and `MDLMPredictor` both pass `total_noise` here.

## 4. Batch contract

`MDLMBatch` ([types_mdlm.py](../../xlm-models/mdlm/types_mdlm.py)):

| Field | Shape | Notes |
|---|---|---|
| `input_ids` | `(B, L)` `int` | Random fraction of tokens replaced by `[MASK]`. |
| `attention_mask` | `(B, L)` `int` | 1 for real tokens, 0 for padding. |
| `target_ids` | `(B, L)` `int` | Original tokens; `-100` at ignored positions when `loss_on_padding=False`. |
| `noise_rate` | `(B,)` `float` | \( \sigma(t) \) — output of `noise_schedule.noise_rate(t)`. |
| `total_noise` | `(B,)` `float` | \( \bar\sigma(t) \) — output of `noise_schedule.total_noise(t)`. |
| `t` | `(B,)` `float` | The sampled time. |

`noise_rate`, `total_noise`, and `t` are produced by `DefaultMDLMCollator` from the wired `NoiseSchedule` (so the collator **requires a real schedule**, not `DummyNoiseSchedule`).

## 5. Loss

`MDLMLoss(loss_on_padding=False, loss_on_visible_tokens=False, model, tokenizer)`:

- `configure(pl_module)` caches `mask_token_id_tensor` on the right device.
- `loss_fn`:
  - Derives `positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)` and zeroes them out at padding.
  - Runs `logits = model(input_ids, total_noise, attention_mask, positions)`.
  - Builds `ignore = (input_ids != mask_token_id)` when `loss_on_visible_tokens=False` (default).
  - `ce = cross_entropy(logits_T, targets, reduction="none", ignore_index=-100)`.
  - **Diffusion weight**: `weight = noise_rate / torch.expm1(total_noise)`; `kl = ce * weight[:, None]`.
  - Returns `loss = masked_mean(kl.flatten(), ~ignore.flatten())`.

## 6. Collators

The internal helper `mdlm_single_segment_collate_fn(examples, noise_schedule, pad_token_id, mask_token_id, ...)` samples `t = noise_schedule.sample_t(batch_size)` and uses `noise_schedule(t)` to compute `(noise_rate, total_noise)`; masks each example with rate `1 - exp(-total_noise)`.

| Class | Input | Output batch | Special behavior |
|---|---|---|---|
| `DefaultMDLMCollator` | `BaseCollatorInput` | `MDLMBatch` | Pad-right to `block_size`, BOS/EOS optional. **Requires a real `NoiseSchedule`.** |
| `MDLMSeq2SeqTrainCollator` | `Seq2SeqCollatorInput` | `MDLMBatch` | Concatenates `[prompt][BOS][target][EOS]` with right padding; masks only suffix positions. |
| `MDLMSeq2SeqPredCollator` | `Seq2SeqCollatorInput` | `MDLMBatch` | `input_ids = left-padded prompt only`; `target_ids = right-padded target` (used for seq2seq prediction). |

Noise schedules live in [noise_mdlm.py](../../xlm-models/mdlm/noise_mdlm.py):

- `ContinuousTimeLinearSchedule(sigma_min, sigma_max)` — affine \(\bar\sigma(t)\) (with the exponential `total_noise`); `t_from_noise_rate` raises `RuntimeError`.
- `ContinuousTimeLogLinearSchedule(sigma_min, sigma_max)` — log-linear total-noise; both `t_from_noise_rate` and `t_from_total_noise` are exact inverses. Requires `sigma_min == 0.0` (raises `NotImplementedError` otherwise).
- Both support `antithetic_sampling=True` (default) — sample `t ~ U[0,1]` then spread via `_convert_to_correlated` (`t / B + arange(B) / B`).
- `grad=True` and `importance_sampling=True` are not implemented and raise.

## 7. Predictor

`MDLMPredictor(max_steps, max_new_tokens=None, tokenizer, model, noise_schedule, top_k=None, top_p=None)`:

- Sampling function selected at `__init__`: top-k -> `sample_from_top_k`, top-p -> `sample_from_top_p`, both -> `sample_categorical`, neither -> `ValueError`.
- `predict()` clones `input_ids`, optionally appends `max_new_tokens` `[MASK]` tokens, derives positions from `attention_mask.cumsum-1`, and starts the diffusion chain at `t = 1`.
- `predict_single_step`:
  - `s = t - dt` where `dt = (1 - 1e-5) / (max_steps + 1)`.
  - `dot_sigma_t, dot_sigma_s = noise_schedule(t)[1], noise_schedule(s)[1]` (i.e. `total_noise`).
  - `chance_t = 1 - exp(-dot_sigma_t)`, `chance_s = 1 - exp(-dot_sigma_s)`.
  - For non-final steps, builds the categorical \(q(x_s | x_t)\): `softmax(logits) * (chance_t - chance_s)` with the mask token bucket set to `chance_s`, then samples `x_s`.
  - For the final step, `argmax(logits)`.
  - Non-mask positions in `x_t` are preserved via `torch.where(masked, x_s, x_t)`.
- `stop()` true when `done.all()` or `t <= 0` everywhere or no `[MASK]` remains.
- Output `MDLMPredictionDict`: `{text, ids, loss=None, time_taken, output_start_idx}`.

## 8. Metrics

See [tests/models/mdlm/test_metrics_mdlm.py](../../tests/models/mdlm/test_metrics_mdlm.py).

| Function | Returned keys |
|---|---|
| `seq2seq_exact_match_update_fn` | `pred = loss_dict["ids"][:, output_start_idx:]`, `target`, `pred_length = pred.shape[-1]`, `target_length` |
| `seq2seq_token_accuracy_update_fn` | `pred`, `target`, `pred_mask = ones_like(pred)` |
| `mean_metric_update_fn` | `value = loss_dict["loss"]` |

## 9. Configs / experiments

Hydra groups under [xlm-models/mdlm/configs/](../../xlm-models/mdlm/configs/). Available experiment entry points:

- `experiment=owt_mdlm` (OpenWebText)
- `experiment=text_mdlm`

## 10. Testing

Tests in [tests/models/mdlm/](../../tests/models/mdlm):

- `test_model_mdlm.py` — extends `BaseModelTests`, plus a positions-from-mask check (added in this plan).
- `test_loss_mdlm.py` — extends `BaseLossTests`.
- `test_collator_mdlm.py` — now uses `real_loglinear_schedule` (added in this plan) to exercise `DefaultMDLMCollator`.
- `test_predictor_mdlm.py` — now uses `real_loglinear_schedule` to exercise `MDLMPredictor.predict()`.
- `test_noise_mdlm.py`, `test_metrics_mdlm.py` — pure-function helpers.

Shared fixtures (`tiny_mdlm_model`, `mdlm_batch`, `simple_tokenizer`, `real_loglinear_schedule`) live in [tests/conftest.py](../../tests/conftest.py) and [tests/models/mdlm/conftest.py](../../tests/models/mdlm/conftest.py).

## 11. API reference

- [`mdlm.model_mdlm`](../reference/mdlm/model_mdlm/)
- [`mdlm.loss_mdlm`](../reference/mdlm/loss_mdlm/)
- [`mdlm.predictor_mdlm`](../reference/mdlm/predictor_mdlm/)
- [`mdlm.datamodule_mdlm`](../reference/mdlm/datamodule_mdlm/)
- [`mdlm.noise_mdlm`](../reference/mdlm/noise_mdlm/)
- [`mdlm.metrics_mdlm`](../reference/mdlm/metrics_mdlm/)
- [`mdlm.types_mdlm`](../reference/mdlm/types_mdlm/)
