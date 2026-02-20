<p align="center">
  <img src="xLM_bold.png" width="600" alt="xLM Logo" />
</p>

<h1 align="center">A Unified Framework for Non-Autoregressive Language Models</h1>


xLM is a modular, research-friendly framework for developing and comparing non-autoregressive language models. Built on PyTorch and PyTorch Lightning, with Hydra for configuration management, xLM makes it effortless to experiment with cutting-edge NAR architectures.

## Key Features


| Feature                                                                    | Description                                                                                                                                                  |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Modular Design**                                                         | Plug-and-play components—swap models, losses, predictors, and collators independently                                                                        |
| **[Lightning-Powered](https://github.com/Lightning-AI/pytorch-lightning)** | Uses PyTorch Lightning for distributed training, mixed precision, and logging out of the box                                                                 |
| **[Hydra Configs](https://hydra.cc/)**                                     | Hierarchical configuration with runtime overrides—no code changes needed                                                                                     |
| **Multiple Architectures**                                                 | Multiple model families ready to use as baselines                                                                                                            |
| **Research-First**                                                         | Lightweight, and type annotated with `jaxtyping`, several debug for quick testing, and flexible code injection points for practially limitless customization |
| **Hub Integration**                                                        | Push trained models directly to Hugging Face Hub                                                                                                             |

## Available Models

| Model  | Full Name                | Description                          |
|--------|--------------------------|--------------------------------------|
| `mlm`  | Masked Language Model    | Classic BERT-style masked prediction |
| `ilm`  | Insertion Language Model | Insertion-based generation           |
| `arlm` | Autoregressive LM        | Standard left-to-right baseline      |
| `mdlm` | Masked Diffusion LM      | Discrete diffusion with masking      |

## Installation

```bash
pip install xlm-core
```

For model implementations, also install:

```bash
pip install xlm-models
```

## Quick Start

xLM uses a simple CLI with three main arguments:

```bash
xlm job_type=<JOB> job_name=<NAME> experiment=<CONFIG>
```

| Argument     | Description                                           |
|--------------|-------------------------------------------------------|
| `job_type`   | One of `prepare_data`, `train`, `eval`, or `generate` |
| `job_name`   | A descriptive name for your run                       |
| `experiment` | Path to your Hydra experiment config                  |

## Next Steps

- [Quick Start](guide/quickstart.md) – Installation, CLI usage, and example workflow
- [API Reference](reference/api.md) – xlm-core and xlm-models API documentation
- [Contributing](guide/contributing.md) – Guidelines for adding new models and features
