<p align="center">
  <img src="./docs/xLM_bold.png" width="180" alt="XLM Logo"/>
</p>

<p align="center">
  <strong>A Unified Framework for Non-Autoregressive Language Models</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/xlm-core/"><img src="https://img.shields.io/pypi/v/xlm-core?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://codecov.io/gh/dhruvdcoder/xlm-core"><img src="https://img.shields.io/codecov/c/github/dhruvdcoder/xlm-core" alt="Code coverage"></a>
  <a href="https://dhruveshp.com/xlm-core/dev/"><img src="https://img.shields.io/badge/Documentation-blue.svg" alt="Documentation"></a>
  <a href="https://github.com/dhruvdcoder/xlm-core"><img src="https://img.shields.io/badge/Python-3.11+-green.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/dhruvdcoder/xlm-core/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

---

XLM is a modular, research-friendly framework for developing and comparing non-autoregressive language models. It is built on PyTorch and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), with [Hydra](https://hydra.cc/) for configuration management.

**Documentation (dev):** [https://dhruveshp.com/xlm-core/dev/](https://dhruveshp.com/xlm-core/dev/)

## Key features

| Feature                    | Description                                                                                                  |
|----------------------------|--------------------------------------------------------------------------------------------------------------|
| **Modular design**         | Plug-and-play components: swap models, losses, predictors, and collators independently.                      |
| **Lightning-powered**      | Distributed training, mixed precision, and logging via PyTorch Lightning.                                    |
| **Hydra configs**          | Hierarchical configuration with runtime overrides.                                                           |
| **Multiple architectures** | Several model families ship in [`xlm-models`](https://github.com/dhruvdcoder/xlm-core/tree/main/xlm-models). |
| **Research-oriented**      | Type annotations (including `jaxtyping`), debug modes, and hooks for metrics and evaluators.                 |
| **Hub integration**        | Push checkpoints to the Hugging Face Hub.                                                                    |

## Installation

```bash
pip install xlm-core
```

For the bundled model implementations in this repository:

```bash
pip install xlm-models
```

Python **3.11+** is required ([`setup.py`](https://github.com/dhruvdcoder/xlm-core/blob/main/setup.py)).

<details>
<summary><strong>Optional extras and contributor setup</strong></summary>

Optional dependency groups (see [`setup.py`](https://github.com/dhruvdcoder/xlm-core/blob/main/setup.py)):

```bash
pip install "xlm-core[safe]"      # SAFE-style molecule preprocessing / evaluators
pip install "xlm-core[molgen]"    # heavier GenMol / Biomemo-related stack
pip install "xlm-core[llm_eval]"  # math-verify / LLM-style benchmarks
pip install "xlm-core[all]"       # union of the above (used in CI)
```

From a git checkout, install in editable mode and pull dev/test/docs/lint stacks as needed:

```bash
pip install -e .
pip install -r requirements/dev_requirements.txt
pip install -r requirements/test_requirements.txt
pip install -r requirements/docs_requirements.txt
pip install -r requirements/lint_requirements.txt
```

Full detail: [Dependencies](https://dhruveshp.com/xlm-core/dev/developers/dependencies/).

</details>

## CLI overview

XLM is driven by Hydra. The usual entrypoint is:

```bash
xlm job_type=<JOB> job_name=<NAME> experiment=<CONFIG>
```

| Argument     | Description                                               |
|--------------|-----------------------------------------------------------|
| `job_type`   | What to run (training, eval, data prep, etc.); see below. |
| `job_name`   | A label for the run.                                      |
| `experiment` | Hydra experiment config (e.g. `lm1b_ilm`).                |

<details>
<summary><strong><code>job_type</code> reference</strong></summary>

| Group                 | `job_type` values                                                                                                                                                                                              |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Main**              | `prepare_data`, `train`, `eval`, `generate`                                                                                                                                                                    |
| **Checkpoints / Hub** | `extract_checkpoint` ([guide](https://dhruveshp.com/xlm-core/dev/guide/extract-checkpoint/)), `push_to_hub` ([guide](https://dhruveshp.com/xlm-core/dev/guide/push-to-hub/))                                   |
| **Hydra helpers**     | `name` (print resolved config tree + job name), `print_predictor_params` (dump predictor config as JSON)                                                                                                       |
| **External models**   | Additional values registered by external packages ([external models](https://dhruveshp.com/xlm-core/dev/guide/external-models/), [custom commands](https://dhruveshp.com/xlm-core/dev/guide/custom-commands/)) |

External commands are dispatched by `job_type` after Hydra loads the config.

</details>

<details>
<summary><strong>Example: ILM on LM1B (full workflow)</strong></summary>

### 1. Prepare data

```bash
xlm job_type=prepare_data job_name=lm1b_prepare experiment=lm1b_ilm
```

### 2. Train

```bash
# Quick debug: overfit a single batch
xlm job_type=train job_name=lm1b_ilm experiment=lm1b_ilm debug=overfit

# Full training
xlm job_type=train job_name=lm1b_ilm experiment=lm1b_ilm
```

### 3. Evaluate

```bash
xlm job_type=eval job_name=lm1b_ilm experiment=lm1b_ilm \
    +eval.ckpt_path=<CHECKPOINT_PATH>
```

### 4. Generate

```bash
xlm job_type=generate job_name=lm1b_ilm experiment=lm1b_ilm \
    +generation.ckpt_path=<CHECKPOINT_PATH>
```

To print generations to the console:

```bash
xlm job_type=generate job_name=lm1b_ilm experiment=lm1b_ilm \
    +generation.ckpt_path=<CHECKPOINT_PATH> \
    debug=[overfit,print_predictions]
```

### 5. Push to the Hugging Face Hub

```bash
xlm job_type=push_to_hub job_name=lm1b_ilm_hub experiment=lm1b_ilm \
    +hub_checkpoint_path=<CHECKPOINT_PATH> \
    +hub.repo_id=<YOUR_REPO_ID>
```

Step-by-step copy of the hosted guide: [Quick Start](https://dhruveshp.com/xlm-core/dev/guide/quickstart/).

</details>

## Model families (`xlm-models`)

The companion package [`xlm-models`](https://github.com/dhruvdcoder/xlm-core/tree/main/xlm-models) registers six top-level families (see [`xlm-models/xlm_models.json`](https://github.com/dhruvdcoder/xlm-core/blob/main/xlm-models/xlm_models.json)). **Documented** means a conceptual guide on the site; **State** is a rough stability hint (the PyPI package as a whole is [beta](https://github.com/dhruvdcoder/xlm-core/blob/main/setup.py)). Cross-family comparison: [Models overview](https://dhruveshp.com/xlm-core/dev/models/).

| Tag       | Name                                                                 | Documented                                              | State | Notes                                                                                                                                                                                        |
|-----------|----------------------------------------------------------------------|---------------------------------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `arlm`    | Autoregressive LM (baseline)                                         | [Full](https://dhruveshp.com/xlm-core/dev/models/arlm/) | Beta  | —                                                                                                                                                                                            |
| `ilm`     | Insertion language model                                             | [Full](https://dhruveshp.com/xlm-core/dev/models/ilm/)  | Beta  | —                                                                                                                                                                                            |
| `mdlm`    | Masked diffusion LM                                                  | [Full](https://dhruveshp.com/xlm-core/dev/models/mdlm/) | Beta  | —                                                                                                                                                                                            |
| `mlm`     | Masked language model (BERT-style)                                   | [Full](https://dhruveshp.com/xlm-core/dev/models/mlm/)  | Beta  | —                                                                                                                                                                                            |
| `flexmdm` | Flexible masked diffusion                                            | Partial                                                 | Alpha | Variable-length / fragment-style masking; [arXiv:2509.01025](https://arxiv.org/abs/2509.01025); [source](https://github.com/dhruvdcoder/xlm-core/tree/main/xlm-models/flexmdm)               |
| `dream`   | Dream-style decoder LM in XLM (`DreamXLMModel`, `DreamPredictor`, …) | Partial                                                 | Alpha | [Source](https://github.com/dhruvdcoder/xlm-core/tree/main/xlm-models/dream); backbone helpers in [`xlm.backbones.dream`](https://dhruveshp.com/xlm-core/dev/reference/xlm/backbones/dream/) |

The API reference includes `xlm` and the four main `xlm-models` families (see [API Reference](https://dhruveshp.com/xlm-core/dev/reference/)).

## Other CLIs

[`setup.py`](https://github.com/dhruvdcoder/xlm-core/blob/main/setup.py) also exposes:

- `xlm-scaffold` — model scaffolding helper
- `xlm-push-to-hub` — dedicated Hub upload entrypoint (in addition to `job_type=push_to_hub`)

## Extending XLM

New architectures generally implement four pieces that plug into the harness:

| Piece         | Role                     |
|---------------|--------------------------|
| **Model**     | Network and forward pass |
| **Loss**      | Training objective       |
| **Predictor** | Inference / generation   |
| **Collator**  | Batch construction       |

Guides:

- [Adding a task or dataset](https://dhruveshp.com/xlm-core/dev/guide/adding-a-task/)
- [Data pipeline](https://dhruveshp.com/xlm-core/dev/guide/data-pipeline/)
- [Metrics](https://dhruveshp.com/xlm-core/dev/guide/metrics/)
- [Evaluate](https://dhruveshp.com/xlm-core/dev/guide/eval/)
- [External models](https://dhruveshp.com/xlm-core/dev/guide/external-models/)
- [Custom commands (`job_type` extensions)](https://dhruveshp.com/xlm-core/dev/guide/custom-commands/)
- [FAQ](https://dhruveshp.com/xlm-core/dev/guide/faq/)

## Developers

- [Contributing](https://dhruveshp.com/xlm-core/dev/guide/CONTRIBUTING/)
- [Running tests](https://dhruveshp.com/xlm-core/dev/developers/testing/running-tests/)
- [Unit tests](https://dhruveshp.com/xlm-core/dev/developers/testing/unit-tests/)
- [Integration tests](https://dhruveshp.com/xlm-core/dev/developers/testing/integration-tests/)

## Project layout

```text
xlm-core/
├── src/xlm/              # Core framework (harness, datamodule, tasks, Hydra configs)
├── xlm-models/           # Model families (arlm, ilm, mlm, mdlm, flexmdm, dream, …)
├── docs/                 # MkDocs source (published as https://dhruveshp.com/xlm-core/dev/)
├── tests/
├── requirements/
└── wiki/                 # Legacy internal notes
```

## Contributing

We welcome contributions. See [CONTRIBUTING.md](./CONTRIBUTING.md) and the [Good First Issue](https://github.com/dhruvdcoder/xlm-core/issues?q=state%3Aopen+label%3A%22good+first+issue%22) list.

## License

This project is licensed under the MIT License.

## Acknowledgements

XLM is developed and maintained by [IESL](https://iesl.cs.umass.edu/) students at UMass Amherst.

**Primary developers**

1. [Dhruvesh Patel](https://dhruveshp.com)
2. [Durga Prasad Maram](https://github.com/Durga-Prasad1)
3. [Sai Sreenivas Chintha](https://github.com/sensai99)
4. [Benjamin Rozonoyer](https://brozonoyer.github.io/)

**External Model Contributors:**
| Contributor    | Model                                            | Paper                                                                                                              |
|----------------|--------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Dhruvesh Patel | [DILM](https://github.com/dhruvdcoder/ctmc_dilm) | [A Continuous Time Markov Chain Framework for Insertion Language Models](https://openreview.net/pdf?id=nCyV21FmUI) |

We welcome external model contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

## Cite

If you found this repository useful, please consider citing:

```bibtex
@article{patel2025xlm,
  title={XLM: A Python package for non-autoregressive language models},
  author={Patel, Dhruvesh and Maram, Durga Prasad and Chintha, Sai Sreenivas and Rozonoyer, Benjamin and McCallum, Andrew},
  journal={arXiv preprint arXiv:2512.17065},
  year={2025}
}
```

---

<p align="center">
  <sub>Built with care for the NLP research community</sub>
</p>
