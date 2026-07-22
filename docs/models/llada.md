# LLaDA — Large Language Diffusion Model

## 1. Overview

`llada` integrates GSAI-ML's [LLaDA](https://arxiv.org/abs/2502.09992) into xLM. It provides the same pieces as the [`dream`](https://github.com/dhruvdcoder/xlm-core/tree/main/xlm-models/dream) package: a backbone whose state-dict keys match the Hub checkpoint, an adapter for the MLM predictor protocol, a diffusion predictor, and Hydra configs for evaluation. Like `dream`, it is currently eval-only — training loss and metrics are not implemented yet.

LLaDA is a masked discrete diffusion LM. A bidirectional Transformer predicts tokens at `[MASK]` positions (`mask_token_id=126336`, string `<|mdm_mask|>`). To generate, the sampler starts from a canvas of mask tokens and, over a number of steps, commits the predictions it is most confident about ("low-confidence remasking" in the paper), optionally working left to right in semi-autoregressive blocks.

```bibtex
@article{nie2025llada,
  title   = {Large Language Diffusion Models},
  author  = {Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal = {arXiv preprint arXiv:2502.09992},
  year    = {2025}
}
```

Both Hub checkpoints share the same architecture, so they use the same model YAML — pick one with `hub.repo_id`:

| Checkpoint | Notes |
|---|---|
| [`GSAI-ML/LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) | Default in `math500_llada_eval` |
| [`GSAI-ML/LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | Override `hub.repo_id` and the tokenizer path (see below) |

Package: {{ gh_dir('xlm-models/llada', 'xlm-models/llada/') }}. Backbone: {{ gh_dir('src/xlm/backbones/llada', 'src/xlm/backbones/llada/') }}.

## 2. Files at a glance

| Module | Public classes / helpers |
|---|---|
| {{ gh('src/xlm/backbones/llada/configuration_llada.py', 'configuration_llada.py') }} | `LLaDAConfigBase`, `ModelConfig`, enums |
| {{ gh('src/xlm/backbones/llada/modeling_llada.py', 'modeling_llada.py') }} | `LLaDAModel`, `LLaDAModelLM`, `LLaDALlamaBlock`, … |
| {{ gh('xlm-models/llada/configuration_llada.py', 'llada/configuration_llada.py') }} | `LLaDAConfig` |
| {{ gh('xlm-models/llada/llada_model.py', 'llada_model.py') }} | `LLaDAHFModel`, `LLaDAXLMModel` |
| {{ gh('xlm-models/llada/predictor_llada.py', 'predictor_llada.py') }} | `LLaDAPredictor` |
| {{ gh('xlm-models/llada/datamodule_llada.py', 'datamodule_llada.py') }} | `print_batch_llada` |

There is no `loss_llada.py` or `metrics_llada.py` yet, since the package is eval-only.

## 3. Architecture

The backbone is ported from the Hub `trust_remote_code` modules with state-dict keys left unchanged, so checkpoints load with `strict=True` and no key remapping. The relevant keys are `model.transformer.wte`, `model.transformer.blocks.{i}.{q,k,v,up,ff}_proj`, `model.transformer.ln_f`, and `model.transformer.ff_out` (the head is untied).

The 8B layout, taken from the Hub `config.json`: `d_model=4096`, 32 layers, 32 MHA heads, `mlp_hidden_size=12288`, `vocab_size=embedding_size=126464`, RoPE with `rope_theta=500000`, no weight tying, and bidirectional attention (`is_causal=False`).

`LLaDAXLMModel` adapts the HF wrapper to the MLM predictor protocol — the same surface as Dream, minus the logits shift:

```python
forward(
    x_t: Tensor,                              # (B, L) input ids
    attention_mask: Optional[Tensor] = None,  # 2D pad mask (native LLaDA path)
    positions: Optional[Tensor] = None,       # RoPE position_ids (packed cumsum OK)
) -> Tensor                                   # (B, L, embedding_size) logits
```

Unlike Dream, LLaDA predicts each masked position in place, so `LogitsShiftBy1` must not be wired in.

## 4. Batch contract

Eval uses the shared MLM seq2seq prediction collator (`MLMSeq2SeqPredCollator`) with MATH-500 fields:

| Field | Shape | Notes |
|---|---|---|
| `input_ids` | `(B, L)` | Left-padded prompt |
| `attention_mask` | `(B, L)` | 1 = real, 0 = pad |
| `answer` / `target` | pass-through | For `Math500Eval` post-hoc |

The predictor then appends `max_new_tokens` copies of `<|mdm_mask|>` (id `126336`) to the right of the prompt; this is the canvas the sampler fills in.

## 5. Loss

Not implemented yet. `model_type/llada_base.yaml` has no `loss:` key, following the same convention as Dream.

## 6. Collators

| Config | Class | Role |
|---|---|---|
| {{ gh('xlm-models/llada/configs/collator/math500_pred_llada.yaml', 'math500_pred_llada') }} | `mlm.datamodule_mlm.MLMSeq2SeqPredCollator` | MATH-500 prompt → prediction batch |

## 7. Predictor

`LLaDAPredictor` subclasses `DreamPredictor` directly — the Dream machinery already implements LLaDA's sampling procedure, so no new sampling code was needed. The knobs map onto the reference sampler (`generate.py` in the LLaDA repo) as follows:

| LLaDA reference | xLM predictor knobs |
|---|---|
| `remasking='low_confidence'` | `confidence: top_prob` |
| `remasking='random'` | `confidence: null` |
| `block_length` | `block_size` (semi-AR blocks) |
| `steps` | `max_steps` |
| `gen_length` | `max_new_tokens` |
| `temperature=0` | `temperature: 0.0` (greedy) |

Classifier-free guidance (`cfg_scale` in the reference code) is not implemented; the reference base-model evals run without it.

## 8. Metrics

Step-level LM metrics are not wired up for LLaDA eval. MATH-500 accuracy is computed after the run by the post-hoc evaluator {{ gh('src/xlm/tasks/math500/__init__.py', 'Math500Eval') }}, which scores the logged predictions with `math_verify`.

## 9. Configs / experiments

Hydra configs under {{ gh_dir('xlm-models/llada/configs', 'xlm-models/llada/configs/') }}:

| Config | Role |
|---|---|
| `model/llada_8b.yaml` | Architecture (Base and Instruct share this file) |
| `model_type/llada_base.yaml` / `llada_eval.yaml` | Harness + predictor (no loss) |
| `experiment/math500_llada_eval.yaml` | MATH-500 Hub eval |
| `datamodule/math500_llada.yaml` | Prediction dataloaders + `print_batch_llada` |
| `debug/math500_debug.yaml` | Tiny debug limits |
| `fsdp/decoder_lm_example.yaml` | Example FSDP grouping for `model.transformer.*` |

The package is registered in `xlm_models.json` (`"llada": "llada"`). Make sure your install can see it:

```bash
# from the xlm-core repo root
pip install -e ./xlm-models
# or: export XLM_MODELS_PACKAGES=mlm:dream:llada
```

### MATH-500 eval (Hub weights)

Prepare the MATH-500 cache once (rank 0):

```bash
xlm job_type=prepare_data job_name=math500_llada_prep \
  experiment=math500_llada_eval num_dataset_workers=4
```

Then run the eval. Weights are pulled from the Hub (`hub.repo_id` in the experiment YAML, `GSAI-ML/LLaDA-8B-Base` by default); you need a GPU with enough memory for the 8B model in bf16 (~16GB+):

```bash
xlm job_type=eval job_name=math500_llada_eval \
  experiment=math500_llada_eval \
  ++trainer.precision=bf16-mixed
```

For the Instruct checkpoint, override the repo id and tokenizer path (the model YAML stays the same):

```bash
xlm job_type=eval job_name=math500_llada_instruct_eval \
  experiment=math500_llada_eval \
  hub.repo_id=GSAI-ML/LLaDA-8B-Instruct \
  global_components.tokenizer.pretrained_model_name_or_path=GSAI-ML/LLaDA-8B-Instruct \
  ++trainer.precision=bf16-mixed
```

!!! note "Hub-only eval and local checkpoints"
    If `checkpointing_dir` already contains `best.ckpt` / `last.ckpt`, those take precedence over Hub weights. Use a fresh `job_name` or a clean `checkpointing_dir` when you want a pure Hub eval. Set `HF_HUB_KEY` (or `.secrets.env`) for the Hub download. See [Evaluate](../guide/eval.md).

For a quick smoke test that only runs a few batches:

```bash
xlm job_type=eval job_name=math500_llada_debug \
  experiment=math500_llada_eval debug=math500_debug \
  ++trainer.precision=bf16-mixed
```

## 10. Testing

{{ gh_dir('tests/models/llada', 'tests/models/llada/') }}:

```bash
# CPU unit tests (tiny config)
pytest tests/models/llada/test_llada_model.py -q

# Full 8B Hub logits parity vs AutoModel(trust_remote_code=True)
XLM_RUN_HUB_TESTS=1 pytest tests/models/llada/test_llada_model.py -k hub
```

## 11. API reference

- Backbone: {{ gh_dir('src/xlm/backbones/llada', 'xlm.backbones.llada') }}
- Package: {{ gh_dir('xlm-models/llada', 'llada') }}
