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
| {{ gh('xlm-models/bd3lm/model_bd3lm.py', 'model_bd3lm.py') }} | `Bd3lmModel`, `DDiTBlock`, `DDiTBlockCausal`, `DDiTFinalLayer`, `Rotary`, `EmbeddingLayer`, `TimestepEmbedder`, `block_diff_mask`, `load_pretrained_bd3lm` |
| {{ gh('xlm-models/bd3lm/loss_bd3lm.py', 'loss_bd3lm.py') }} | `Bd3lmLoss` |
| {{ gh('xlm-models/bd3lm/predictor_bd3lm.py', 'predictor_bd3lm.py') }} | `Bd3lmPredictor`, `Bd3lmUnconditionalPredictor` |
| {{ gh('xlm-models/bd3lm/datamodule_bd3lm.py', 'datamodule_bd3lm.py') }} | `DefaultBd3lmCollator`, `Bd3lmSeq2SeqCollator`, `Bd3lmSeq2SeqPredCollator`, `Bd3lmUnconditionalPredCollator`, `Bd3lmEmptyDataset`, `print_batch_bd3lm` |
| {{ gh('xlm-models/bd3lm/noise_schedule.py', 'noise_schedule.py') }} | `Bd3lmNoise`, `LogLinearNoise`, `CosineNoise`, `ExpNoise`, `LogarithmicNoise` |
| {{ gh('xlm-models/bd3lm/metrics_bd3lm.py', 'metrics_bd3lm.py') }} | `seq2seq_exact_match_update_fn`, `seq2seq_token_accuracy_update_fn`, `mean_metric_update_fn`, `perplexity_metric_update_fn` |
| {{ gh('xlm-models/bd3lm/types_bd3lm.py', 'types_bd3lm.py') }} | `Bd3lmBatch`, `Bd3lmSeq2SeqBatch`, `Bd3lmLossDict`, `Bd3lmPredictionDict`, `Bd3lmModel` (Protocol) |
| {{ gh('xlm-models/bd3lm/convert_hf_checkpoint.py', 'convert_hf_checkpoint.py') }} | `load_released_state_dict`, `strip_backbone_prefix` |

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
| `target_ids` | `(B, L)` seq / `(B, T)` s2s | the clean tokens |
| `loss_scale` | `(B, L)` | from the noise schedule |
| `sigma` | `(B, 1)` | per-example noise level |


## 5. Loss

`Bd3lmLoss` computes the diffusion NLL over the masked positions.

`loss_on_padding` … controls whether answer-side PAD takes part …


## 6. Collators

| Config | Class | Role |
|---|---|---|
| {{ gh('xlm-models/bd3lm/configs/collator/default_bd3lm.yaml', 'default_bd3lm') }} | `DefaultBd3lmCollator` | pre-training: noises the whole sequence |
| {{ gh('xlm-models/bd3lm/configs/collator/seq2seq_bd3lm.yaml', 'seq2seq_bd3lm') }} | `Bd3lmSeq2SeqCollator` | Supervised Seq2Seq training  |
| {{ gh('xlm-models/bd3lm/configs/collator/seq2seq_pred_bd3lm.yaml', 'seq2seq_pred_bd3lm') }} | `Bd3lmSeq2SeqPredCollator` | Seq2Seq Generation |
| {{ gh('xlm-models/bd3lm/configs/collator/unconditional_pred_bd3lm.yaml', 'unconditional_pred_bd3lm') }} | `Bd3lmUnconditionalPredCollator` | unconditional generation |



## 7. Predictor

`Bd3lmPredictor` implements the semi-autoregressive sampler: blocks are generated left to
right, and within each block one position is unmasked per step.

| Key (under `model.config.sampling`) | Default | Meaning |
|---|---|---|
| `confidence_decoding` | `true` | unmask the most confident position |
| `first_hitting` | `true` | first-hitting sampler (Zheng et al., 2025) |
| `var_length` | `true` | stop at EOS instead of filling the window |
| `nucleus_p` | `0.9` | nucleus Sampling |
| `kv_cache` | `true` | enables KV-caching |

Setting `confidence_decoding=false` gives the reference implementation's uniformly random
unmasking.

`Bd3lmUnconditionalPredictor` generates with no prompt: the sampler draws a prior block
and writes BOS at position 0. It is driven by `Bd3lmEmptyDataset`, which supplies blank
rows through xLM's `UnconditionalGenerationDatasetManager`.

## 8. Metrics

| Metric | Where |
|---|---|
| `accumulated_loss` | train / val / test, both model_types |
| `perplexity` | val / test, `bd3lm_unconditional` only |
| `exact_match`, `token_accuracy` | val / test prediction, `bd3lm` (seq2seq star-graph) |

For unconditional *sample* quality rather than modelling quality, use xLM's post-hoc
generative perplexity evaluator, e.g. `post_hoc_evaluator=gen_ppl_gpt2_large`.

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
| `pretrained/{auto,owt_bs4,owt_bs8,owt_bs16}.yaml` | to use the pretrained HuggingFace models |

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

### Fine-tuning from a released checkpoint

```bash
xlm job_type=train job_name=my_finetune \
  experiment=star_medium_bd3lm \
  +pretrained=auto
```

The checkpoint is derived from `block_size`, so `block_size=8` pulls
`kuleshov-group/bd3lm-owt-block_size8`. Weights are fetched at model construction and
cached by HuggingFace. Compatible checkpoints are `block_size` 4, 8 and 16, plus
`block_size1024-pretrain` (pin that one with `+pretrained_from=<repo>`); all are
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
