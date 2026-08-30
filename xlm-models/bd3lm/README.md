# BD3-LM

An xLM implementation of the **Block Discrete Denoising Diffusion Language Model**
(BD3-LM), based on the reference implementation of Arriola et al. (2025).

Supports:

- **unconditional pre-training** 
- **supervised seq2seq training**
- **fine-tuning from released checkpoints**

```
bd3lm/
├── model_bd3lm.py           
├── loss_bd3lm.py            
├── predictor_bd3lm.py       
├── datamodule_bd3lm.py     
├── metrics_bd3lm.py         
├── noise_schedule.py        
├── types_bd3lm.py           
└── configs/
    ├── model/bd3lm{,_tiny,_small,_medium}.yaml   
    ├── model_type/bd3lm.yaml                     
    ├── model_type/bd3lm_unconditional.yaml       
    ├── collator/{default,unconditional_pred,seq2seq,seq2seq_pred}_bd3lm.yaml
    ├── datamodule/{owt,star{,_easy,_medium,_hard}}_bd3lm*.yaml
    ├── experiment/{owt,star_{easy,medium,hard}}_bd3lm*.yaml
    ├── pretrained/kuleshov_group_bd3lm.yaml                        # released checkpoints from the Hub
    ├── datasets/bd3lm_empty_pred.yaml            
    ├── metrics/perplexity_bd3lm.yaml
    └── noise_schedule/bd3lm.yaml
```

## Quickstart

Pick the experiment; it selects the dataset, the collator and the metrics.

**Unconditional pre-training** on OpenWebText:

```bash
xlm job_type=train job_name=my_run experiment=owt_bd3lm
```

**Seq2seq** on the star-graph path-finding task, in three difficulties:

| experiment | dataset | prompt / target | inference config |
|---|---|---|---|
| `star_easy_bd3lm` | `dhruveshpatel/star-small` | 28 / 12 | `star_easy_bd3lm_inference` |
| `star_medium_bd3lm` | `dhruveshpatel/star-medium` | 36 / 12 | `star_medium_bd3lm_inference` |
| `star_hard_bd3lm` | `dhruveshpatel/star-hard` | 116 / 24 | `star_hard_bd3lm_inference` |

```bash
xlm job_type=train job_name=my_run experiment=star_medium_bd3lm
```

Evaluate a checkpoint with the matching `_inference` config:

```bash
xlm job_type=eval job_name=my_eval \
  experiment=star_medium_bd3lm_inference \
  eval.ckpt_path=/path/to/last.ckpt
```

## Inference

Supports confidence-based decoding and random unmasking.

```bash
# confidence-based (default)
xlm ... model.config.sampling.confidence_decoding=true \
        model.config.sampling.confidence=prob_diff   # or top_prob, entropy

# random, as in the reference implementation
xlm ... model.config.sampling.confidence_decoding=false
```

## Unconditional generation

Generate from a released checkpoint. `+pretrained=kuleshov_group_bd3lm` picks the one matching your
`block_size` and pulls it from the Hub:

```bash
xlm job_type=eval job_name=my_gen experiment=owt_bd3lm_inference +pretrained=kuleshov_group_bd3lm
```

## Model sizes

All three model size variants from the reference block diffusion implementation are
available — `tiny`, `small` and `medium`. Pick one with `model=`:

```bash
xlm job_type=train job_name=my_run \
  experiment=star_medium_bd3lm \
  model=bd3lm_tiny        # or bd3lm_small (default), bd3lm_medium
```

## Cite

If you use this model in your research, please cite the original paper along with xLM.

```bibtex
@inproceedings{arriola2025block,
      title={Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models},
      author={Marianne Arriola and Aaron Gokaslan and Justin T Chiu and Zhihan Yang and Zhixuan Qi and Jiaqi Han and Subham Sekhar Sahoo and Volodymyr Kuleshov},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
      url={https://arxiv.org/abs/2503.09573},
}

@article{patel2025xlm,
  title={XLM: A Python package for non-autoregressive language models},
  author={Patel, Dhruvesh and Maram, Durga Prasad and Chintha, Sai Sreenivas and Rozonoyer, Benjamin and McCallum, Andrew},
  journal={arXiv preprint arXiv:2512.17065},
  year={2025}
}
