# External Language Models for xLM

xLM supports external language models that can be developed and maintained separately from the core framework. This allows researchers to keep their model code clean and self-contained, and share models without including the entire xLM codebase. The code follows a modular design with four main components that work together to provide a complete language modeling solution. You need to implement all of them in order to add a new working model.

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

## Main Components

### 1. **LossFunction**
The `LossFunction` is responsible for computing the training loss during model training, validation and optionally test time.

**Key Responsibilities:**
- Compute loss between model predictions and ground truth targets
- Return a dictionary with "loss" key and any other additional values that you want to track.

**Interface:**
```python
class LossFunction(Generic[T_in, T_out], Protocol):
    model: Any
    tokenizer: Tokenizer

    def loss_fn(self, batch: T_in, ...) -> T_out: ...
    def configure(self, pl_module: "Harness"): ...
        """Converts scalar to tensor such that loss_fn becomes compile friendly. If you don't want to compile you don't need to implement this.
        """
```

Examples: `xlm-models/arlm/loss_arlm.py`

### 2. **Predictor**
The `Predictor` handles generating output sequences from the model.

**Key Responsibilities:**
- Run the model (typically in a loop) to produce a sequence of tokens.
- Convert generated token_ids to text.

**Interface:**
```python
class Predictor(Generic[T_in, T_out_pred], Protocol):
    tokenizer: Tokenizer
    noise_schedule: NoiseSchedule
    model: Any

    def predict(self, batch: T_in, ...) -> T_out_pred: ...
    def to_dict(self, batch: T_in, preds: T_out_pred, ...) -> List[Dict[str, Any]]: ...
```

Examples: `xlm-models/arlm/predictor_arlm.py`

### 3. **Collator**
The `Collator` prepares batches of data for training and inference. It handles data preprocessing, padding, and batching.

**Key Responsibilities:**
- It receives raw token_ids and converts them to a dict which is passed in as a batch to the model.
- Handle padding and truncation.

**Interface:**
```python
class Collator(Protocol):
    """For pre-training the model on language modeling."""
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]: ...

class Seq2SeqCollator(Collator):
    """For training the model on seq2seq tasks."""
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]: ...

class Seq2SeqCollatorPrediction(Collator):
    """For generating predictions for seq2seq tasks."""
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]: ...
```

Examples: `xlm-models/arlm/datamodule_arlm.py`

### 4. **Model**
The `Model` is the bare neural network architecture for your LM. It defines the forward pass and model parameters.

**Key Responsibilities:**
- Define the neural network architecture.
- Implement the forward pass.

**Interface:**
```python
class Model:
    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, ...) -> Tensor: ...
```

Examples: `xlm-models/arlm/model_arlm.py`

All these four components are designed to be aware of each other, and are only expected to run with each other for the same LM and not with any other LM. This is a key design choice that allows one to implement really esoteric models without worrying about how to abstract them such that their dataflow becomes compatible with other LMs.

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
│   └── commands.yaml                # Optional: see [Custom Commands](custom-commands.md)
├── setup.py                         # Package installation (optional)
└── README.md                        # Documentation
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

Key config locations (paths may vary for external models in `xlm-models/`):

- `configs/lightning_train/collator/` – Collator configs
- `configs/lightning_train/datamodule/` – Datamodule configs
- `configs/lightning_train/model/` – Model (neural network) configs
- `configs/lightning_train/model_type/` – Loss, predictor, metrics
- `configs/lightning_train/experiment/` – Experiment configs

## Discovery Methods

xLM discovers external models through two approaches:

### 1. Directory-Based Discovery

Place your model directory in one of these locations:

- Current directory (`.`)
- `xlm-models/` directory
- Directory specified by the `XLM_MODELS_PATH` environment variable

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

Install your model as a Python package and register it via the `XLM_MODELS_PACKAGES` environment variable (colon-separated list of installed package names).

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

**Solution:** Each model must have a unique name. Check for duplicate entries in `xlm_models.json` files or conflicts with core xLM models.

### Config Not Found

```
hydra.errors.ConfigCompositionException: Could not find 'model/my_model'
```

**Check:**

- `configs/model/my_model.yaml` exists
- `configs/model_type/my_model.yaml` exists
- Config files have valid YAML syntax
- `_target_` paths point to correct classes

### Hydra Errors

If you see `Unable to find or instantiate abc.xyz.MyClass`, try importing manually: `python -c "from abc.xyz import MyClass"`.

### Unable to implement a component

Check existing models in `xlm-models/` (arlm, mlm, ilm, mdlm) for reference.

## Examples

See the core models in the `xlm-models` package for complete examples:

- `arlm` - Auto-regressive language model
- `mlm` - Masked language model
- `ilm` - Infilling language model
- `mdlm` - Masked diffusion language model
