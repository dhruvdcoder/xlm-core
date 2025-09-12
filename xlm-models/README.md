# XLM Models

This directory contains all language models for the XLM framework. Each model is implemented as an independent Python package that can be installed separately.

## Available Models

| Model    | Description                        | Status         |
|----------|------------------------------------|----------------|
| `arlm`   | Auto-Regressive Language Model     | ✅ Migrated     |
| `idlm`   | Iterative Diffusion Language Model | ✅ Migrated     |
| `ilm`    | Infilling Language Model           | ✅ Migrated     |
| `mlm`    | Masked Language Model              | ✅ Migrated     |
| `mdlm`   | Masked Diffusion Language Model    | ✅ Migrated     |
| `elm`    | Edit Language Model                | ✅ Migrated     |
| `indigo` | Indigo Model                       | ✅ Migrated     |
| `zlm`    | Zero Language Model                | 📁 Placeholder |

## Installation

### Install All Models
```bash
pip install xlm-models[all]
```

### Install Specific Models
```bash
# Install individual models
pip install xlm-models[arlm]
pip install xlm-models[idlm,mlm]

# Or install from subdirectories
pip install ./arlm
pip install ./idlm
```

### Development Installation
```bash
# Install in development mode
pip install -e ./arlm
```

## Usage

After installation, models can be used in XLM configs:

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

When adding a new model:
1. Create the model directory structure
2. Implement all required components
3. Add the model name to `.xlm_models`
4. Update the main `setup.py` extras_require
5. Test installation and functionality

## Migration Status

This directory represents the migration of models from `src/xlm/lm/` to independent packages. All models that were previously part of the core XLM package are being moved here to create a cleaner separation between the framework and model implementations.
