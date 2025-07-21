# External Language Models for XLM

XLM supports external language models that can be developed and maintained separately from the core framework. This allows researchers to:

- Keep their model code clean and self-contained
- Share models without including the entire XLM codebase

## Overview

External models work through a plugin system that:
1. **Discovers models** declared in a `.xlm_models` file
2. **Validates** model structure and dependencies (whether all required files are present)
3. **Registers** Python packages that should be imported for the model to work
4. **Extends** Hydra config search paths to include the model's config files

## Quick Start

### 1. Scaffold a New Model

Use the scaffolding script to create a complete model structure:

```bash
xlm-scaffold my_awesome_model
```

This creates scaffolding for the model which includes:
- A complete Python package with skeleton implementation for `model.py`, `loss.py`, `predictor.py`, `collators.py`, `metrics.py` and `types.py`
- All necessary Hydra configuration files for the model
- Adds model to `.xlm_models` file (this is a simple text file that lists the models to load)

### 2. Implement Your Model

The scaffolded files contain detailed TODOs and docstrings. Key files to implement:


- `my_awesome_model/types.py` - Type definitions that used across all the other files
- `my_awesome_model/model.py` - Neural network architecture
- `my_awesome_model/loss.py` - Loss computation
- `my_awesome_model/predictor.py` - Inference/generation logic
- `my_awesome_model/collators.py` - Data preprocessing
- `my_awesome_model/metrics.py` - Metrics computation

### 3. Test Your Model

```bash
cd my_awesome_model
xlm job_type=train \
  job_name=my_model_test \
  experiment=my_awesome_model_debug \
  debug=overfit
```

## External Model Structure

Each external model follows this structure:

```
my_awesome_model/                    # External model directory
├── my_awesome_model/                # Python package
│   ├── __init__.py                 # Package exports
│   ├── types.py                    # Type definitions
│   ├── model.py                    # Neural network
│   ├── loss.py                     # Loss function
│   ├── predictor.py                # Inference logic
│   ├── collators.py                # Data preprocessing
│   └── metrics.py                  # Metrics computation
├── configs/                        # Hydra configurations
│   ├── model/
│   │   └── my_awesome_model.yaml  # Model config
│   ├── model_type/
│   │   └── my_awesome_model.yaml  # Model type config
│   ├── collator/
│   │   ├── default_my_awesome_model.yaml
│   │   └── seq2seq_my_awesome_model.yaml
│   └── experiment/
│       └── my_awesome_model_debug.yaml
├── setup.py                       # Package installation
└── README.md                      # Documentation
```

## Required Components

### 1. Python Package Components

#### **Types** (`types.py`)
Define data structures using `TypedDict`:

```python
class MyModelBatch(TypedDict):
    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]

class MyModelLossDict(TypedDict):
    loss: Float[TT, ""]
    batch_loss: Float[TT, " batch"]

class MyModelPredictionDict(TypedDict):
    text: List[str]
    ids: Integer[TT, " batch seq_len"]
```

#### **Model** (`model.py`)
Neural network implementation:

```python
class MyAwesomeModel(nn.Module):
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Implement your architecture
        return logits
```

#### **Loss Function** (`loss.py`)
Training loss computation:

```python
class MyAwesomeLoss(LossFunction[MyModelBatch, MyModelLossDict]):
    def loss_fn(self, batch, **kwargs):
        # Compute loss from model outputs
        return {"loss": loss, "batch_loss": batch_loss}
```

#### **Predictor** (`predictor.py`)
Inference and generation:

```python
class MyAwesomePredictor(Predictor[MyModelBatch, MyModelPredictionDict]):
    def predict(self, batch, **kwargs):
        # Generate predictions
        return {"text": generated_text, "ids": generated_ids}
```

#### **Collators** (`collators.py`)
Data preprocessing and batching:

```python
class DefaultMyAwesomeCollator(Collator):
    def __call__(self, examples):
        # Process and batch data
        return batched_data
```

### 2. Configuration Files

#### **Model Config** (`configs/model/my_awesome_model.yaml`)
```yaml
# @package _global_
model:
  _target_: my_awesome_model.model.MyAwesomeModel
  num_embeddings: ${tokenizer:full_vocab_size}
  d_model: 768
  # ... other model parameters

tags:
  model: my_awesome_model
```

#### **Model Type Config** (`configs/model_type/my_awesome_model.yaml`)
```yaml
# @package _global_
lightning_module:
  _target_: xlm.harness.Harness

loss:
  _target_: my_awesome_model.loss.MyAwesomeLoss

predictor:
  _target_: my_awesome_model.predictor.MyAwesomePredictor
  # ... predictor parameters

reported_metrics:
  train:
    lm:
      accumulated_loss:
        update_fn: my_awesome_model.metrics.mean_metric_update_fn
  # ... other metrics

tags:
  model_type: my_awesome_model
```

## The `.xlm_models` File

The `.xlm_models` file declares which external models to load:

```
# External models for this project
my_awesome_model
another_model

# Comments are supported
# model_under_development  # Commented out models are ignored
```

**Rules:**
- One model name per line
- Comments start with `#`
- Empty lines are ignored
- Model names must match directory names

## Development Workflow

### 1. **Scaffolding**
```bash
# Create model structure
xlm-scaffold my_model

# Check generated files
cd my_model
ls -la
```

### 2. **Implement**

1. 
```bash
# Edit core components
my_model/types.py      # Define types
my_model/model.py      # Implement architecture
my_model/loss.py       # Implement loss
my_model/predictor.py  # Implement generation
my_model/collators.py  # Implement data processing
```

2. Edit the config files

### 3. **Test**
Quickly test your model on the synthetic dataset `star_easy` dataset.
```bash
# Quick overfit test
xlm job_type=train \
  job_name=star_easy_my_model \
  experiment=star_easy_my_model \
  debug=overfit
```



## Validation and Conflict Resolution

The system automatically validates external models and detects conflicts:

### **Model Name Conflicts**
```
Error: Duplicate model names found in .xlm_models: ['my_model']
```
**Solution:** Use unique names for each model

### **Python Package Conflicts**
```
Error: Python package conflict: Both 'model_a' and 'model_b' define package 'model'
```
**Solution:** Use model-specific package names

### **Missing Required Configs**
```
Error: External model 'my_model' missing required config groups: ['model_type']
```
**Solution:** Create all required configuration files

### **Config Validation**
Enable strict validation:
```python
from xlm.external_models import setup_external_models
setup_external_models(strict_validation=True)
```

## Best Practices

### **Naming Conventions**
- Use lowercase with underscores: `my_awesome_model`
- Avoid conflicts with core models: `arlm`, `ilm`, `mlm`

### **Code Organization**
- Keep implementations focused and minimal
- Use type hints and docstrings extensively
- Follow XLM patterns from existing models
- Add TODO comments for incomplete sections

### **Configuration**
- Start with debug configs for fast iteration
- Use reasonable default parameters
- Document all configuration options
- Test with different datasets

### **Testing Strategy**
1. **Unit testing**: Test individual components
2. **Overfit testing**: Verify model can memorize small data
3. **Small data**: Test on reduced datasets
4. **Integration**: Full training pipeline
5. **Evaluation**: Model performance metrics

## Advanced Features

### **Multiple Models Support**
```
# .xlm_models
transformer_variant_b
custom_architecture
```

### **Environment Variables**
```bash
# Search additional directories
export XLM_MODELS_PATH="/path/to/external/models"

# Use different models file
xlm-scaffold my_model --xlm-models-file custom_models.txt
```

### **Package Installation**
```bash
# Install external model as Python package
cd my_awesome_model
pip install -e .

# Now importable from anywhere
python -c "from my_awesome_model import MyAwesomeModel"
```

### **Sharing Models**
```bash
# Create distributable package
cd my_awesome_model
python setup.py sdist bdist_wheel

# Share via PyPI, GitHub, etc.
pip install my-awesome-model
```

## Integration with Core XLM

External models integrate seamlessly with XLM's training infrastructure:

- **Lightning Integration**: Automatic integration with PyTorch Lightning
- **Hydra Configuration**: Full Hydra config composition support  
- **Metrics**: Integration with XLM's metrics framework
- **Logging**: Automatic logging to wandb, tensorboard
- **Checkpointing**: Standard checkpoint saving/loading
- **Multi-GPU**: Automatic DDP/FSDP support

## Troubleshooting

### **Import Errors**
```
ModuleNotFoundError: No module named 'my_model'
```
**Check:**
- Model listed in `.xlm_models`
- Directory exists and has correct structure
- Python package has `__init__.py`

### **Config Errors**
```
hydra.errors.ConfigCompositionException: Missing config group 'model_type'
```
**Check:**
- All required config files exist
- Config files have correct YAML syntax
- Target paths point to correct classes

### **Validation Errors**
```
ExternalModelConflictError: Duplicate model names found
```
**Check:**
- No duplicate names in `.xlm_models`
- No conflicts with core XLM models
- Unique package names

### **Runtime Errors**
```
NotImplementedError: Forward pass not implemented yet
```
**Check:**
- All TODO sections are implemented
- Required methods have actual implementations
- Type signatures match expected interfaces

## Examples

See the `zlm` model in this repository for a complete example of:
- External model structure
- Configuration setup
- Integration with XLM training
- Testing and validation

## Migration from Core Models

To convert a core XLM model to external:

1. **Copy model code** to external directory
2. **Update imports** to be self-contained
3. **Create configs** following external model patterns
4. **Update targets** to point to external classes
5. **Add to `.xlm_models`** file
6. **Test integration** with debug configs

## Contributing

When contributing external models:

1. **Follow naming conventions** and best practices
2. **Include comprehensive tests** and documentation
3. **Validate** with the external model system
4. **Provide examples** and usage instructions
5. **Share configuration** for reproducibility

## Support

For questions about external models:
- Check existing models for examples
- Review error messages and validation output
- Use debug configs for troubleshooting
- Reach out on Slack for guidance

The external model system is designed to be flexible and extensible. Happy modeling! 🚀 