# MLM - Masked Language Model

> **Migration Note**: This model has been migrated from `xlm.lm.mlm` to be an independent external package. 
> All import paths have been updated from `xlm.lm.mlm.*` → `mlm.*` and config targets updated accordingly.
> The functionality remains exactly the same.

## Installation

```bash
# Install from xlm-models
pip install xlm-models[mlm]

# Or install directly
pip install ./mlm

# Development installation
pip install -e ./mlm
```

## Usage with New Import Paths

```yaml
# Model configuration (updated target path)
model:
  _target_: mlm.model_mlm.RotaryTransformerMLMModel

# Loss configuration (updated target path)  
loss:
  _target_: mlm.loss_mlm.MLMLoss

# Predictor configuration (updated target path)
predictor:
  _target_: mlm.predictor_mlm.MLMPredictor
```

## Components

### Model Architecture (`model_mlm.py`)
- `RotaryTransformerMLMModel`: Transformer model for masked language modeling

### Loss Function (`loss_mlm.py`)
- `MLMLoss`: Masked language modeling loss function

### Predictor (`predictor_mlm.py`)
- `MLMPredictor`: Text generation with masking strategies

### Data Processing (`datamodule_mlm.py`)
- `DefaultMLMCollator`: Standard MLM collator
- `MLMSeq2SeqTrainCollator`: Training collator for seq2seq tasks
- `MLMSeq2SeqCollator`: Sequence-to-sequence collator
- `MLMSeq2SeqPredCollator`: Prediction collator

### History Tracking (`history_mlm.py`)
- `HistoryTopKPlugin`: Tracks top-k predictions during generation

### Types (`types_mlm.py`)
- Type definitions for batches, predictions, and model interfaces


# Commands

```bash
# Sudoku (debug overfit)
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=sudoku_mlm" "experiment=sudoku_mlm" "debug=overfit"
```