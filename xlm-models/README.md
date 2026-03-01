# XLM Models

This directory contains all language models for the XLM framework. All models are bundled together in a single `xlm-models` package.

## Available Models

| Model  | Description                     | Status     |
|--------|---------------------------------|------------|
| `arlm` | Auto-Regressive Language Model  | ✅ Migrated |
| `ilm`  | Infilling Language Model        | ✅ Migrated |
| `mlm`  | Masked Language Model           | ✅ Migrated |
| `mdlm` | Masked Diffusion Language Model | ✅ Migrated |


## Installation

All models are bundled in a single `xlm-models` package.

```bash
# Install from local repo (all models included)
pip install -e ./xlm-models

# Or from PyPI once published (all models included)
pip install xlm-models
```

**Note:** All models are always installed. There is no selective installation - you get everything in one package.

## Usage

After installation, models can be imported directly:

```python
# Import from any model
from mlm import RotaryTransformerMLMModel, MLMLoss, MLMPredictor
from arlm import RotaryTransformerARLMModel, ARLMLoss, ARLMPredictor
from mdlm import MDLMModel, MDLMLoss, MDLMPredictor
# etc.
```

Or used in Hydra configs:

```yaml
# Model configuration
model:
  _target_: arlm.model_arlm.RotaryTransformerARLMModel
  
# Model type configuration  
model_type: arlm
```

## Model Structure

Each model follows this structure:
```
model_name/
├── model_name/           # Python package
│   ├── __init__.py
│   ├── types_model.py    # Type definitions
│   ├── model_model.py    # Neural network
│   ├── loss_model.py     # Loss function
│   ├── predictor_model.py # Inference logic
│   ├── datamodule_model.py # Data processing
│   └── metrics_model.py  # Metrics computation
├── configs/              # Hydra configurations
│   ├── model/
│   ├── model_type/
│   ├── collator/
│   └── experiment/
├── setup.py             # Package installation
└── README.md           # Model documentation
```

## Development

### Adding a Model to xlm-models Package

To contribute a new model to the bundled `xlm-models` package:

1. Create the model directory structure (e.g., `newmodel/newmodel/`)
2. Implement all required components (`__init__.py`, model files, etc.)
3. Add the model name to `.xlm_models`
4. Update the main `xlm-models/setup.py`:
   - Add model name to the `packages` list
   - Add mapping to `package_dir` dict (e.g., `"newmodel": "newmodel/newmodel"`)
   - Update the model list in `long_description`
5. Test installation and functionality

Example for adding a model called `newmodel`:
```python
# In xlm-models/setup.py
packages=["arlm", "idlm", "ilm", "mlm", "mdlm", "elm", "indigo", "newmodel"],
package_dir={
    "arlm": "arlm/arlm",
    "ilm": "ilm/ilm",
    "mlm": "mlm/mlm",
    "mdlm": "mdlm/mdlm",
    "newmodel": "newmodel/newmodel",
}
```

### Creating a Standalone Model Package

To create your own independent model package (for personal use or separate release):

1. Create your model directory structure following the same pattern:
   ```
   mymodel/
   ├── mymodel/              # Python package
   │   ├── __init__.py
   │   ├── model_mymodel.py
   │   ├── loss_mymodel.py
   │   └── ...
   ├── configs/              # Hydra configurations
   ├── setup.py              # Package installation
   └── README.md
   ```

2. Create a `setup.py` in your model directory (see existing models for reference)

3. Install independently:
   ```bash
   pip install -e ./mymodel
   ```

4. Your model can be used without being part of `xlm-models`:
   ```python
   from mymodel import MyModelClass
   ```

This approach allows you to develop and release your model independently without modifying the main `xlm-models` package.

## Migration Status

This directory represents the migration of models from `src/xlm/lm/` to independent packages. All models that were previously part of the core XLM package are being moved here to create a cleaner separation between the framework and model implementations.
