# External Language Models for XLM

XLM supports external language models that can be developed and maintained separately from the core framework. This allows researchers to keep their model code clean and self-contained, and share models without including the entire XLM codebase.

## Quick Start

### 1. Scaffold a New Model

Use the scaffolding script to create a complete model structure:

```bash
xlm-scaffold my_awesome_model
```

This creates:

- A complete Python package with skeleton implementations
- All necessary Hydra configuration files
- Registers the model in `xlm_models.json`

### 2. Implement Your Model

The scaffolded files contain detailed TODOs and docstrings. Key files to implement:

- `my_awesome_model/types_my_awesome_model.py` - Type definitions
- `my_awesome_model/model_my_awesome_model.py` - Neural network architecture
- `my_awesome_model/loss_my_awesome_model.py` - Loss computation
- `my_awesome_model/predictor_my_awesome_model.py` - Inference/generation logic
- `my_awesome_model/datamodule_my_awesome_model.py` - Data preprocessing
- `my_awesome_model/metrics_my_awesome_model.py` - Metrics computation

### 3. Test Your Model

```bash
xlm job_type=train \
  job_name=my_model_test \
  experiment=star_easy_my_awesome_model \
  debug=overfit
```

## Model Structure

Each external model follows this structure:

```
my_awesome_model/                    # Model root directory
├── my_awesome_model/                # Python package
│   ├── __init__.py
│   ├── types_my_awesome_model.py
│   ├── model_my_awesome_model.py
│   ├── loss_my_awesome_model.py
│   ├── predictor_my_awesome_model.py
│   ├── datamodule_my_awesome_model.py
│   └── metrics_my_awesome_model.py
├── configs/                         # Hydra configurations
│   ├── model/
│   ├── model_type/
│   ├── collator/
│   ├── datamodule/
│   ├── experiment/
│   └── commands.yaml                # Optional: custom commands
├── setup.py                         # Package installation (optional)
└── README.md                        # Documentation
```

## Discovery Methods

XLM discovers external models through two approaches:

### 1. Directory-Based Discovery

Place your model directory in one of these locations:

- Current directory (`.`)
- `xlm-models/` directory
- Directory specified by `XLM_MODELS_PATH` environment variable

Create a `xlm_models.json` file in the search directory:

```json
{
  "my_awesome_model": "my_awesome_model",
  "another_model": "path/to/another_model"
}
```

The paths are relative to the directory containing `xlm_models.json`.

**Example:**

```bash
# Project structure
.
├── xlm_models.json          # {"my_model": "my_model"}
├── my_model/
│   ├── my_model/            # Python package
│   └── configs/
└── ...
```

### 2. Package-Based Discovery

Install your model as a Python package and register it via the `XLM_MODELS_PACKAGES` environment variable.

**Package structure requirements:**

Each model needs its own `setup.py` that packages the configs:

```python
# setup.py
setup(
    name="my_awesome_model",
    packages=["my_awesome_model"],
    package_data={
        "my_awesome_model": ["configs/**/*.yaml", "configs/**/*.yml"],
    },
    include_package_data=True,
)
```

**Installation and registration:**

Each model must be installed independently:

```bash
# Install first model
pip install -e ./my_awesome_model

# Install second model (separate setup.py)
pip install -e ./another_model

# Register both installed packages
export XLM_MODELS_PACKAGES="my_awesome_model:another_model"
```

**Core models** (`arlm`, `mlm`, `ilm`, `mdlm`) are automatically discovered and don't need to be added to `XLM_MODELS_PACKAGES`.

## Custom Commands

Models can define custom commands that extend XLM's CLI by creating `configs/commands.yaml`:

```yaml
# configs/commands.yaml
my_custom_command: "my_awesome_model.commands.my_function"
preprocess_data: "my_awesome_model.commands.preprocess"
```

Usage:

```bash
xlm command=my_custom_command arg1=value1 arg2=value2
```

The command functions should accept an `omegaconf.DictConfig` parameter containing the Hydra configuration.

## Environment Variables

- **`XLM_MODELS_PATH`**: Additional directory to search for `xlm_models.json` files

  ```bash
  export XLM_MODELS_PATH="/path/to/external/models"
  ```

- **`XLM_MODELS_PACKAGES`**: Colon-separated list of installed Python packages to discover

  ```bash
  export XLM_MODELS_PACKAGES="my_model1:my_model2"
  ```

## Configuration

External models integrate with Hydra's configuration system. Use them in your experiments:

```yaml
# configs/experiment/my_experiment.yaml
defaults:
  - override /model: my_awesome_model
  - override /model_type: my_awesome_model
  - override /datamodule: star_easy_my_awesome_model
```

Or via command line:

```bash
xlm job_type=train model=my_awesome_model model_type=my_awesome_model
```

## Troubleshooting

### Model Not Found

```
Error: No module named 'my_model'
```

**Check:**

- Model is listed in `xlm_models.json` (directory-based)
- Model is installed and listed in `XLM_MODELS_PACKAGES` (package-based)
- Directory structure is correct
- `__init__.py` exists in the Python package

### Duplicate Model Names

```
ExternalModelConflictError: Duplicate model name: my_model
```

**Solution:** Each model must have a unique name. Check for duplicate entries in `xlm_models.json` files or conflicts with core XLM models.

### Config Not Found

```
hydra.errors.ConfigCompositionException: Could not find 'model/my_model'
```

**Check:**

- `configs/model/my_model.yaml` exists
- `configs/model_type/my_model.yaml` exists
- Config files have valid YAML syntax
- `_target_` paths point to correct classes

## Examples

See the core models in the `xlm-models` package for complete examples:

- `arlm` - Auto-regressive language model
- `mlm` - Masked language model
- `ilm` - Infilling language model
- `mdlm` - Masked diffusion language model
