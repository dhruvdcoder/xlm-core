# LLaDA — Large Language Diffusion Model

## 1. Overview

`llada` integrates [LLaDA](https://arxiv.org/abs/2502.09992) (GSAI-ML) into xLM at **Dream parity**: a Hub-key-compatible backbone, an MLM-protocol adapter, a diffusion predictor, and Hydra eval configs. Training loss / metrics are **not** implemented yet (Alpha / eval-focused), same status as [`dream`](https://github.com/dhruvdcoder/xlm-core/tree/main/xlm-models/dream).

LLaDA is a masked discrete diffusion LM: a bidirectional Transformer predicts tokens at `[MASK]` positions (`mask_token_id=126336`, string `<|mdm_mask|>`). Sampling fills a fixed canvas of masks and commits the most confident predictions per semi-autoregressive block (low-confidence remasking).

```bibtex
@article{nie2025llada,
  title   = {Large Language Diffusion Models},
  author  = {Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal = {arXiv preprint arXiv:2502.09992},
  year    = {2025}
}
```

Hub checkpoints (same architecture; switch via `hub.repo_id`):

| Checkpoint | Notes |
|---|---|
| [`GSAI-ML/LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) | Default in `math500_llada_eval` |
| [`GSAI-ML/LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | Same model YAML; override `hub.repo_id` + tokenizer path |

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

No `loss_llada.py` / `metrics_llada.py` yet (eval-only).

## 3. Architecture

Ported from the Hub `trust_remote_code` modules with **state-dict keys unchanged**, so weights load with `strict=True` and no remapping:

`model.transformer.wte`, `model.transformer.blocks.{i}.{q,k,v,up,ff}_proj`, `model.transformer.ln_f`, `model.transformer.ff_out` (untied).

8B layout (from Hub `config.json`): `d_model=4096`, 32 layers, 32 MHA heads, `mlp_hidden_size=12288`, `vocab_size=embedding_size=126464`, RoPE (`rope_theta=500000`), `weight_tying=false`, bidirectional attention (`is_causal=False` / MDM path).

`LLaDAXLMModel` adapts the HF wrapper to the MLM predictor protocol (same surface as Dream, without a logits shift):

```python
forward(
    x_t: Tensor,                              # (B, L) input ids
    attention_mask: Optional[Tensor] = None,  # 2D pad mask (native LLaDA path)
    positions: Optional[Tensor] = None,       # RoPE position_ids (packed cumsum OK)
) -> Tensor                                   # (B, L, embedding_size) logits
```

Unlike Dream, LLaDA predicts each masked position **in place** — do **not** wire `LogitsShiftBy1`.

## 4. Batch contract

Eval uses the shared MLM seq2seq prediction collator (`MLMSeq2SeqPredCollator`) with MATH-500 fields:

| Field | Shape | Notes |
|---|---|---|
| `input_ids` | `(B, L)` | Left-padded prompt |
| `attention_mask` | `(B, L)` | 1 = real, 0 = pad |
| `answer` / `target` | pass-through | For `Math500Eval` post-hoc |

Predictor appends `max_new_tokens` of `<|mdm_mask|>` (id `126336`) as the generation canvas.

## 5. Loss

Not implemented. There is no `loss:` key under `model_type/llada_base.yaml` (Dream-style).

## 6. Collators

| Config | Class | Role |
|---|---|---|
| {{ gh('xlm-models/llada/configs/collator/math500_pred_llada.yaml', 'math500_pred_llada') }} | `mlm.datamodule_mlm.MLMSeq2SeqPredCollator` | MATH-500 prompt → prediction batch |

## 7. Predictor

`LLaDAPredictor` subclasses `DreamPredictor`. Mapping to the LLaDA reference sampler (`generate.py`):

| LLaDA reference | xLM predictor knobs |
|---|---|
| `remasking='low_confidence'` | `confidence: top_prob` |
| `remasking='random'` | `confidence: null` |
| `block_length` | `block_size` (semi-AR blocks) |
| `steps` | `max_steps` |
| `gen_length` | `max_new_tokens` |
| `temperature=0` | `temperature: 0.0` (greedy) |

Classifier-free guidance (`cfg_scale`) from the reference code is **not** implemented.

## 8. Metrics

Step-level LM metrics are not wired for LLaDA eval. MATH-500 accuracy comes from the post-hoc evaluator {{ gh('src/xlm/tasks/math500/__init__.py', 'Math500Eval') }} over logged predictions.

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

Register the package (`xlm_models.json` already lists `"llada": "llada"`). Ensure the editable install / `XLM_MODELS_PACKAGES` can see it:

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

Run eval (loads `GSAI-ML/LLaDA-8B-Base` via `hub.repo_id` in the experiment YAML). Use a GPU with enough memory for bf16 8B (~16GB+):

```bash
xlm job_type=eval job_name=math500_llada_eval \
  experiment=math500_llada_eval \
  ++trainer.precision=bf16-mixed
```

Instruct checkpoint (same model YAML):

```bash
xlm job_type=eval job_name=math500_llada_instruct_eval \
  experiment=math500_llada_eval \
  hub.repo_id=GSAI-ML/LLaDA-8B-Instruct \
  global_components.tokenizer.pretrained_model_name_or_path=GSAI-ML/LLaDA-8B-Instruct \
  ++trainer.precision=bf16-mixed
```

!!! note "Hub-only eval and local checkpoints"
    If `checkpointing_dir` already contains `best.ckpt` / `last.ckpt`, those **override** Hub weights. Use a fresh `job_name` or a clean `checkpointing_dir`. Set `HF_HUB_KEY` (or `.secrets.env`) for Hub download. See [Evaluate](../guide/eval.md).

Debug smoke (few batches):

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
