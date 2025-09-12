# ARLM - Auto-Regressive Language Model

ARLM (Auto-Regressive Language Model) is a transformer-based language model implementation for the XLM framework.

## Overview

ARLM implements standard causal language modeling where the model predicts the next token given the previous tokens. It features:

- **Rotary Position Embeddings**: Uses rotary embeddings for better position encoding
- **Causal Attention**: Standard left-to-right attention pattern
- **Flexible Architecture**: Configurable layers, heads, and dimensions
- **Seq2Seq Support**: Can be used for both language modeling and sequence-to-sequence tasks

## Installation

```bash
# Install from xlm-models
pip install xlm-models[arlm]

# Or install directly
pip install ./arlm

# Development installation
pip install -e ./arlm
```

## Components

### Model Architecture (`model_arlm.py`)
- `RotaryTransformerARLMModel`: Main transformer model with rotary embeddings

### Loss Function (`loss_arlm.py`)
- `ARLMLoss`: Cross-entropy loss for causal language modeling

### Predictor (`predictor_arlm.py`)
- `ARLMPredictor`: Auto-regressive text generation with various sampling strategies

### Data Processing (`datamodule_arlm.py`)
- `DefaultARLMCollator`: Standard collator for language modeling
- `ARLMSeq2SeqCollator`: Collator for sequence-to-sequence tasks
- `ARLMSeq2SeqPredCollator`: Collator for prediction tasks

### Types (`types_arlm.py`)
- Type definitions for batches, predictions, and model interfaces

## Usage

### Basic Configuration

```yaml
# Model configuration
model:
  _target_: arlm.model_arlm.RotaryTransformerARLMModel
  num_embeddings: ${tokenizer:full_vocab_size}
  d_model: 768
  num_layers: 12
  nhead: 12

# Model type configuration
model_type: arlm

# Use ARLM loss and predictor
loss:
  _target_: arlm.loss_arlm.ARLMLoss

predictor:
  _target_: arlm.predictor_arlm.ARLMPredictor
```

### Training

```bash
# Train on synthetic data
xlm job_type=train \
    model=rotary_transformer_small_arlm \
    model_type=arlm \
    experiment=star_easy_arlm \
    debug=overfit
```

## Available Configs

### Models
- `rotary_transformer_small_arlm.yaml`: Small transformer configuration

### Model Types  
- `arlm_seq2seq.yaml`: Sequence-to-sequence configuration

### Collators
- `default_arlm.yaml`: Standard language modeling collator
- `seq2seq_arlm.yaml`: Sequence-to-sequence collator
- `seq2seq_pred_arlm.yaml`: Prediction collator

### Datamodules
- `star_easy_arlm.yaml`: STAR easy dataset
- `star_medium_arlm.yaml`: STAR medium dataset  
- `star_hard_arlm.yaml`: STAR hard dataset
- `owt_arlm.yaml`: OpenWebText dataset
- `lm1b_arlm.yaml`: Language Model 1 Billion dataset

### Experiments
- `star_easy_arlm.yaml`: Complete experiment configuration for STAR easy

## Migration Notes

This package was migrated from `xlm.lm.arlm` to be an independent external model. The main changes:

1. **Import paths**: `xlm.lm.arlm.*` → `arlm.*`
2. **Config targets**: Updated to point to the new package structure
3. **Package structure**: Now installable as a separate package

All functionality remains the same, but the model is now completely independent from the core XLM framework.

## Development

When modifying ARLM:

1. **Python files**: Located in `arlm/`
2. **Configs**: Located in `configs/`
3. **Tests**: Add tests for any new functionality
4. **Documentation**: Update this README for significant changes

## Examples

See the `configs/experiment/` directory for complete usage examples.
