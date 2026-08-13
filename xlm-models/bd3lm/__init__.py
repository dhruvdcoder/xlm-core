"""
Bd3lm - External Language Model for XLM Framework

This package implements the Bd3lm model with all necessary components:
- Model architecture (model_bd3lm.py)
- Loss function (loss_bd3lm.py) 
- Predictor for inference (predictor_bd3lm.py)
- Data module (datamodule_bd3lm.py)
- Metrics computation (metrics_bd3lm.py)
- Type definitions (types_bd3lm.py)

To use this model:
1. Add 'bd3lm' to your xlm_models.json file
2. Use model_type=bd3lm and model=bd3lm in your config
"""

from .model_bd3lm import Bd3lmModel
from .loss_bd3lm import Bd3lmLoss
from .predictor_bd3lm import Bd3lmPredictor
from .datamodule_bd3lm import DefaultBd3lmCollator, Bd3lmSeq2SeqCollator
from .types_bd3lm import (
    Bd3lmBatch,
    Bd3lmSeq2SeqBatch, 
    Bd3lmLossDict,
    Bd3lmPredictionDict,
)

__all__ = [
    "Bd3lmModel",
    "Bd3lmLoss", 
    "Bd3lmPredictor",
    "DefaultBd3lmCollator",
    "Bd3lmSeq2SeqCollator",
    "Bd3lmBatch",
    "Bd3lmSeq2SeqBatch",
    "Bd3lmLossDict",
    "Bd3lmPredictionDict",
]
