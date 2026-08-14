# BD3-LM — Block Discrete Denoising Diffusion Language Model

## 1. Overview

`bd3lm` integrates [BD3-LM](https://arxiv.org/abs/2503.09573) into xLM. The backbone is a
DDiT-style Transformer with rotary positional embeddings.

It supports:

- unconditional pre-training
- supervised seq2seq training
- fine-tuning from the released BD3-LM checkpoints on HuggingFace

```bibtex
@inproceedings{arriola2025block,
  title     = {Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models},
  author    = {Marianne Arriola and Aaron Gokaslan and Justin T Chiu and Zhihan Yang and Zhixuan Qi and Jiaqi Han and Subham Sekhar Sahoo and Volodymyr Kuleshov},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://arxiv.org/abs/2503.09573}
}
```

Package: {{ gh_dir('xlm-models/bd3lm', 'xlm-models/bd3lm/') }}. See
{{ gh('xlm-models/bd3lm/README.md', 'xlm-models/bd3lm/README.md') }}.

## 2. Files at a glance

| Module | Public classes / helpers |
|---|---|
| {{ gh('xlm-models/bd3lm/model_bd3lm.py', 'model_bd3lm.py') }} | `Bd3lmModel`, `DDiTBlock`, `DDiTBlockCausal`, `DDiTFinalLayer`, `Rotary`, `EmbeddingLayer`, `TimestepEmbedder`, `block_diff_mask`, `DDiT` |
| {{ gh('xlm-models/bd3lm/loss_bd3lm.py', 'loss_bd3lm.py') }} | `Bd3lmLoss` |
| {{ gh('xlm-models/bd3lm/predictor_bd3lm.py', 'predictor_bd3lm.py') }} | `Bd3lmPredictor`, `Bd3lmUnconditionalPredictor` |
| {{ gh('xlm-models/bd3lm/datamodule_bd3lm.py', 'datamodule_bd3lm.py') }} | `DefaultBd3lmCollator`, `Bd3lmSeq2SeqCollator`, `Bd3lmSeq2SeqPredCollator`, `Bd3lmUnconditionalPredCollator`, `Bd3lmEmptyDataset`, `print_batch_bd3lm` |
| {{ gh('xlm-models/bd3lm/noise_schedule.py', 'noise_schedule.py') }} | `Bd3lmNoise`, `LogLinearNoise`, `CosineNoise`, `ExpNoise`, `LogarithmicNoise` |
| {{ gh('xlm-models/bd3lm/metrics_bd3lm.py', 'metrics_bd3lm.py') }} | `seq2seq_exact_match_update_fn`, `seq2seq_token_accuracy_update_fn`, `mean_metric_update_fn`, `perplexity_metric_update_fn` |
| {{ gh('xlm-models/bd3lm/types_bd3lm.py', 'types_bd3lm.py') }} | `Bd3lmBatch`, `Bd3lmSeq2SeqBatch`, `Bd3lmLossDict`, `Bd3lmPredictionDict`, `Bd3lmModel` (Protocol) |

## 3. Architecture

A DDiT-style Transformer with rotary embeddings and a block-causal attention mask.
Three sizes from the reference implementation are available:

| `model=` | hidden | blocks | heads | cond_dim |
|---|---|---|---|---|
| `bd3lm_tiny` | 256 | 8 | 8 | 64 |
| `bd3lm_small` (= `bd3lm`, default) | 768 | 12 | 12 | 128 |
| `bd3lm_medium` | 1024 | 24 | 16 | 128 |

`model.config.model.length` is `prompt_size + target_size`, and `attn_backend` defaults
to `sdpa` so the model runs without flash-attn.

```python
forward(
    indices: Tensor,                          # (B, 2L) cat(x_t, x_0) in training
    sigma: Optional[Tensor],                  # (B,) noise level; zeroed when
                                              #   algo.time_conditioning=false
    attention_mask: Optional[Tensor] = None,  # (B, L); None when nothing is padded
    positions: Optional[Tensor] = None,       # (B, L); None means plain arange
    sample_mode: bool = False,
    store_kv: bool = False,
) -> Tensor                                   # (B, L, vocab_size)
```

`indices` is the noisy and clean sequences concatenated, which is why it is twice the
length in training: each noisy block attends to the clean tokens of the blocks before it.
Only the noisy half is returned as logits.

## 4. Batch contract

Both training collators emit:

| Field | Shape | Notes |
|---|---|---|
| `x0` | `(B, L)` | clean sequence |
| `xt` | `(B, L)` | noised sequence |
| `input_ids` | `(B, 2L)` | `cat(xt, x0)`, what the model consumes |
| `attention_mask` | `(B, L)` | 1 = real, 0 = pad |
| `loss_mask` | `(B, L)` | 1 where the position is `[MASK]` and should be scored |
| `target_ids` | `(B, L)` seq / `(B, T)` s2s | the clean tokens — **unshifted**, no `-100` |
| `loss_scale` | `(B, L)` | from the noise schedule |
| `sigma` | `(B, 1)` | per-example noise level |

Diffusion denoises in place, so there is no shift-by-one target. In the seq2seq case
`target_ids` is answer-width only, and the loss slices the last `target_len` positions to
match it.

## 5. Loss

`Bd3lmLoss` computes the diffusion NLL over the masked positions.

`loss_on_padding` (declared once in the model_type config and read by both the loss and
the collator) controls whether answer-side PAD takes part in diffusion and in the loss:

- `true` — PAD is treated like any other token; `q_xt` may noise it into `[MASK]`, and
  those positions are scored.
- `false` — PAD is protected from noising and excluded from the loss, so only real
  tokens produce gradient.

## 6. Collators

| Config | Class | Role |
|---|---|---|
| {{ gh('xlm-models/bd3lm/configs/collator/default_bd3lm.yaml', 'default_bd3lm') }} | `DefaultBd3lmCollator` | pre-training: noises the whole sequence |
| {{ gh('xlm-models/bd3lm/configs/collator/seq2seq_bd3lm.yaml', 'seq2seq_bd3lm') }} | `Bd3lmSeq2SeqCollator` | keeps the prompt clean, noises the answer |
| {{ gh('xlm-models/bd3lm/configs/collator/seq2seq_pred_bd3lm.yaml', 'seq2seq_pred_bd3lm') }} | `Bd3lmSeq2SeqPredCollator` | prompt → prediction batch |
| {{ gh('xlm-models/bd3lm/configs/collator/unconditional_pred_bd3lm.yaml', 'unconditional_pred_bd3lm') }} | `Bd3lmUnconditionalPredCollator` | all-`[MASK]` canvas for unconditional generation |

`ignore_bos` keeps the BOS at position 0 out of the diffusion — never noised, never
scored — mirroring `algo.ignore_bos` in the reference implementation.

## 7. Predictor

`Bd3lmPredictor` implements the semi-autoregressive sampler: blocks are generated left to
right, and within each block one position is unmasked per step.

| Key (under `model.config.sampling`) | Default | Meaning |
|---|---|---|
| `confidence_decoding` | `true` | unmask the most confident position rather than a random one |
| `confidence` | `prob_diff` | scoring criterion: `top_prob`, `prob_diff`, `entropy` |
| `first_hitting` | `true` | first-hitting sampler (Zheng et al., 2025) |
| `var_length` | `true` | stop at EOS instead of filling the window |
| `nucleus_p` | `0.9` | nucleus filtering within a block |
| `kv_cache` | `false` | cache finalised blocks instead of re-encoding the prefix |

Setting `confidence_decoding=false` gives the reference implementation's uniformly random
unmasking.

`kv_cache=true` produces byte-identical generations to the uncached path. It is only
sound because the sampling mask is strictly block-causal (prompt tokens never attend to
the answer) and `algo.time_conditioning=false` makes the prompt's hidden states constant
across denoising steps — **turning `time_conditioning` on invalidates it**. It also
requires `prompt_len % block_size == 0`, which the code checks.

`Bd3lmUnconditionalPredictor` generates with no prompt: the sampler draws a prior block
and writes BOS at position 0. It is driven by `Bd3lmEmptyDataset`, which supplies blank
rows through xLM's `UnconditionalGenerationDatasetManager`.

## 8. Metrics

| Metric | Where |
|---|---|
| `accumulated_loss` | train / val / test, both model_types |
| `perplexity` | val / test, `bd3lm_unconditional` only |
| `exact_match`, `token_accuracy` | val / test prediction, `bd3lm` (seq2seq) only |

For unconditional *sample* quality rather than modelling quality, use
`experiment=owt_bd3lm_inference`, which wires up xLM's post-hoc generative perplexity
evaluator against GPT-2 Large. See [Generative perplexity](#generative-perplexity-released-checkpoint).

## 9. Configs / experiments

Hydra configs under {{ gh_dir('xlm-models/bd3lm/configs', 'xlm-models/bd3lm/configs/') }}:

| Config | Role |
|---|---|
| `model/bd3lm{,_tiny,_small,_medium}.yaml` | architecture + algo + sampling |
| `model_type/bd3lm.yaml` | seq2seq: loss, predictor, exact-match metrics |
| `model_type/bd3lm_unconditional.yaml` | loss + perplexity, unconditional predictor |
| `experiment/star_{easy,medium,hard}_bd3lm.yaml` | seq2seq training |
| `experiment/star_{easy,medium,hard}_bd3lm_inference.yaml` | matching eval configs |
| `experiment/owt_bd3lm.yaml` | unconditional pre-training on OpenWebText |
| `experiment/owt_bd3lm_inference.yaml` | unconditional generation + generative perplexity |
| `datamodule/owt_bd3lm_pred.yaml` | generation-only datamodule (downloads nothing) |
| `pretrained/auto.yaml` | released checkpoint from the Hub, chosen by `block_size` |
| `datasets/bd3lm_empty_pred.yaml` | blank rows for unconditional generation |

The package is registered in `xlm_models.json` (`"bd3lm": "bd3lm"`).

### Seq2seq training

Star-graph path finding, in three difficulties:

| experiment | dataset | prompt / target |
|---|---|---|
| `star_easy_bd3lm` | `dhruveshpatel/star-small` | 28 / 12 |
| `star_medium_bd3lm` | `dhruveshpatel/star-medium` | 36 / 12 |
| `star_hard_bd3lm` | `dhruveshpatel/star-hard` | 116 / 24 |

```bash
xlm job_type=train job_name=my_run experiment=star_medium_bd3lm
```

Evaluate with the matching `_inference` config:

```bash
xlm job_type=eval job_name=my_eval \
  experiment=star_medium_bd3lm_inference \
  eval.ckpt_path=/path/to/last.ckpt
```

### Unconditional pre-training

```bash
xlm job_type=train job_name=my_run experiment=owt_bd3lm
```

### Unconditional generation

`owt_bd3lm_inference` generates from an empty prompt and scores the samples with GPT-2
Large. It pins the reference implementation's sampling protocol, so the numbers are
comparable to the paper out of the box.

```bash
xlm job_type=eval job_name=my_gen experiment=owt_bd3lm_inference +pretrained=auto
```

`+pretrained=auto` sets `hub.repo_id` from `block_size`, so the weights come through
xLM's shared checkpoint loader - the same path LLaDA and Dream use. `Bd3lmModel` holds
the transformer as `self.backbone`, so the released tensor names match ours 1:1 and
nothing needs converting first.

This experiment swaps in `datamodule/owt_bd3lm_pred.yaml`, which declares only the
prediction managers. `owt_bd3lm` also declares train/val/test `lm` managers over
`owt-gpt2-1024-split`, and `prepare_data()` takes no stage argument — so evaluating with
the training experiment would download and tokenise the 26GB train split before
generating anything.

### Fine-tuning from a released checkpoint

```bash
xlm job_type=train job_name=my_finetune \
  experiment=star_medium_bd3lm \
  +pretrained=auto
```

The checkpoint is derived from `block_size`, so `block_size=8` pulls
`kuleshov-group/bd3lm-owt-block_size8`. Weights are fetched at model construction and
cached by HuggingFace. Compatible checkpoints are `block_size` 4, 8 and 16, plus
`block_size1024-pretrain` (pin that one with `hub.repo_id=<repo>`); all are
768 / 12 / 12 on the GPT-2 vocabulary.

On a task whose vocabulary is not GPT-2's, the three vocabulary-sized tensors cannot
transfer — they are skipped and reported, and the transformer blocks still load:

```
[bd3lm] warm start from kuleshov-group/bd3lm-owt-block_size4: loaded 128/131 tensors
[bd3lm]   3 skipped on shape (training from scratch) - usually a vocabulary difference
```

!!! note "Newer released repos are not compatible"
    The `{owt,gsm8k,cnndm}-bd3lm-s*` repos are a later release with a different config
    schema and a Qwen3 backbone rather than this DiT. They will not load here.

## 10. Results

| experiment | exact match | token accuracy |
|---|---|---|
| `star_easy` (`dhruveshpatel/star-small`) | 1.000 | 1.000 |
| `star_medium` (`dhruveshpatel/star-medium`) | 1.000 | 1.000 |

Verified at batch sizes 1, 4 and 16, and on star-medium additionally with
`kv_cache=true`.

`star-medium` has **22.6% train/test overlap** (1,101 of 4,872 unique test examples also
appear in train), so an unseen-only figure should be reported alongside. `star-hard` is
not reported — its answer lengths vary far more, and block diffusion is expected to find
it about as hard as an autoregressive model does.

### Generative perplexity, released checkpoint

`kuleshov-group/bd3lm-owt-block_size4` loaded through `owt_bd3lm_inference`, generating
at length 1024 and scored by GPT-2 Large:

| | samples | Gen. PPL |
|---|---|---|
| **this implementation** | 300 | **25.74** |
| reference, Table 7 | 300 | 25.70 |

Mean sample length 1020.1 tokens, entropy 5.29.

Sampling matches `scripts/gen_ppl/genppl_bd3lm.sh` in the reference repo: batch size 1,
`nucleus_p=0.9`, `first_hitting=true`, `var_length=false`, `kv_cache=true`, `algo.T=5000`,
and `confidence_decoding=false` so the sampler unmasks at random as the reference does.
All 131 tensors load under `strict=True`. Generation took ~1h35 on one A100.

!!! note "Differences from the reference measurement"
    Ours is a single seed; the paper pools its 300 samples over 12. Scoring also differs
    slightly: xLM masks padding via the attention mask, while the reference sets
    `pad_token = eos_token` and so has to drop *every* `<|endoftext|>` from the average,
    including real document separators. The 0.04 gap suggests neither matters much here.

## 11. Testing

No unit tests yet. The package is exercised through the eval configs above.
