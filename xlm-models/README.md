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

### Install Specific Models
```bash
# Install individual models from subdirectories
pip install ./xlm-models/arlm
pip install ./xlm-models/idlm
pip install ./xlm-models/mlm
```

### Install All Models
```bash
# Install all models at once
pip install ./xlm-models/arlm ./xlm-models/idlm ./xlm-models/ilm ./xlm-models/mlm ./xlm-models/mdlm ./xlm-models/elm ./xlm-models/indigo
```

### Development Installation
```bash
# Install in development mode (from project root)
pip install -e ./xlm-models/arlm
pip install -e ./xlm-models/mdlm

# Or if you're already in the xlm-models directory
cd xlm-models
pip install -e ./arlm
pip install -e ./mdlm
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
4. Create a `setup.py` for the model package
5. Test installation and functionality

## Migration Status

This directory represents the migration of models from `src/xlm/lm/` to independent packages. All models that were previously part of the core XLM package are being moved here to create a cleaner separation between the framework and model implementations.
