<p align="center">
  <img src="./docs/xLM_bold.png" width="180" alt="XLM Logo"/>
</p>

<!-- <h1 align="center">XLM</h1> -->

<p align="center">
  <strong>A Unified Framework for Non-Autoregressive Language Models</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/xlm-core/"><img src="https://img.shields.io/pypi/v/xlm-core?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://github.com/dhruvdcoder/xlm-core"><img src="https://img.shields.io/badge/Python-3.11+-green.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/dhruvdcoder/xlm-core/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

---

XLM is a **modular, research-friendly framework** for developing and comparing non-autoregressive language models. Built on PyTorch and PyTorch Lightning, with Hydra for configuration management, XLM makes it effortless to experiment with cutting-edge NAR architectures.

## ✨ Key Features

| Feature                       | Description                                                                           |
|-------------------------------|---------------------------------------------------------------------------------------|
| 🧩 **Modular Design**         | Plug-and-play components—swap models, losses, predictors, and collators independently |
| ⚡ **Lightning-Powered**       | Distributed training, mixed precision, and logging out of the box                     |
| 🎛️ **Hydra Configs**          | Hierarchical configuration with runtime overrides—no code changes needed              |
| 📦 **Multiple Architectures** | 7 NAR model families ready to use                                                     |
| 🔬 **Research-First**         | Type-safe with `jaxtyping`, debug modes, and flexible metric injection                |
| 🤗 **Hub Integration**        | Push trained models directly to Hugging Face Hub                                      |

## 🏗️ Available Models

| Model  | Full Name                | Description                          |
|--------|--------------------------|--------------------------------------|
| `mlm`  | Masked Language Model    | Classic BERT-style masked prediction |
| `ilm`  | Insertion Language Model | Iterative insertion-based generation |
| `arlm` | Autoregressive LM        | Standard left-to-right baseline      |
| `mdlm` | Masked Diffusion LM      | Discrete diffusion with masking      |
| `idlm` | Diffusion Insertion LM   | Multi-token insertion diffusion      |

## 🚀 Installation

```bash
pip install xlm-core
```

For model implementations, also install:

```bash
pip install xlm-models
```

## 📖 Quick Start

XLM uses a simple CLI with three main arguments:

```bash
xlm job_type=<JOB> job_name=<NAME> experiment=<CONFIG>
```

| Argument     | Description                                           |
|--------------|-------------------------------------------------------|
| `job_type`   | One of `prepare_data`, `train`, `eval`, or `generate` |
| `job_name`   | A descriptive name for your run                       |
| `experiment` | Path to your Hydra experiment config                  |

## 🎯 Example: ILM on LM1B

A complete workflow demonstrating the Insertion Language Model on the LM1B dataset:

### 1️⃣ Prepare Data

```bash
xlm job_type=prepare_data job_name=lm1b_prepare experiment=lm1b_ilm
```

### 2️⃣ Train

```bash
# Quick debug run (overfit a single batch)
xlm job_type=train job_name=lm1b_ilm experiment=lm1b_ilm debug=overfit

# Full training
xlm job_type=train job_name=lm1b_ilm experiment=lm1b_ilm
```

### 3️⃣ Evaluate

```bash
xlm job_type=eval job_name=lm1b_ilm experiment=lm1b_ilm \
    +eval.ckpt_path=<CHECKPOINT_PATH>
```

### 4️⃣ Generate

```bash
xlm job_type=generate job_name=lm1b_ilm experiment=lm1b_ilm \
    +generation.ckpt_path=<CHECKPOINT_PATH>
```

**Tip:** Add `debug=[overfit,print_predictions]` to print generated samples to the console:

```bash
xlm job_type=generate job_name=lm1b_ilm experiment=lm1b_ilm \
    +generation.ckpt_path=<CHECKPOINT_PATH> \
    debug=[overfit,print_predictions]
```

### 5️⃣ Push to Hugging Face Hub

```bash
xlm job_type=push_to_hub job_name=lm1b_ilm_hub experiment=lm1b_ilm \
    +hub_checkpoint_path=<CHECKPOINT_PATH> \
    +hub.repo_id=<YOUR_REPO_ID>
```

## 🗂️ Project Structure

```
xlm-core/
├── src/xlm/           # Core framework
│   ├── harness.py     # PyTorch Lightning module
│   ├── datamodule.py  # Data loading & collation
│   ├── metrics.py     # Evaluation metrics
│   └── configs/       # Default Hydra configs
│
└── xlm-models/        # Model implementations
    ├── mlm/           # Masked LM
    ├── ilm/           # Infilling LM
    ├── arlm/          # Autoregressive LM
    └── ...            # Other architectures
```

## 🔧 Extending XLM

Adding a new model requires implementing four components:

| Component     | Responsibility              |
|---------------|-----------------------------|
| **Model**     | Neural network architecture |
| **Loss**      | Training objective          |
| **Predictor** | Inference/generation logic  |
| **Collator**  | Batch preparation           |


You can also add new entrypoint scripts to the cli.

See the [Contributing Guide](./wiki/CONTRIBUTING.md) for a complete walkthrough.

## 📚 Documentation

- [Data Pipeline](./wiki/datapipeline.md) – How data flows through XLM
- [Training Scripts](./wiki/scripts/training.md) – Advanced training options
- [Generation](./wiki/scripts/generation.md) – Decoding strategies and parameters
- [External Models](./wiki/EXTERNAL_MODELS.md) – Using pretrained weights

## 🤝 Contributing

We welcome model contributions! Please check out our [Contributing Guide](./wiki/CONTRIBUTING.md) for guidelines on adding new models and features.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

XLM is developed and maintained by [IESL](https://iesl.cs.umass.edu/) students at UMass Amherst.

**Primary Developers:**

1. [Dhruvesh Patel](https://dhruveshp.com) 
2. [Durga Prasad Maram](https://github.com/Durga-Prasad1)
3. [Sai Sreenivas Chintha](https://github.com/sensai99) 
4. [Benjamin Rozonoyer](https://brozonoyer.github.io/)

**Model Contributors:**
1. Soumitra Das (EditFlow)
2. Eric Chen (EditFlow)

---

<p align="center">
  <sub>Built with ❤️ for the NLP research community</sub>
</p>
