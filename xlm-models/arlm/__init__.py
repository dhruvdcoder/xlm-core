"""
ARLM - Auto-Regressive Language Model for XLM Framework

This package implements the ARLM model with all necessary components:
- Model architecture (model_arlm.py)
- Loss function (loss_arlm.py)
- Predictor for inference (predictor_arlm.py)
- Data module (datamodule_arlm.py)
- Metrics computation (metrics_arlm.py)
- Type definitions (types_arlm.py)

This model was migrated from xlm.lm.arlm to be an external model.
"""

from .model_arlm import RotaryTransformerARLMModel
from .loss_arlm import ARLMLoss
from .predictor_arlm import ARLMPredictor
from .datamodule_arlm import (
    DefaultARLMCollator,
    ARLMSeq2SeqCollator,
    ARLMSeq2SeqPredCollator,
)
from .types_arlm import (
    ARLMBatch,
    ARLMSeq2SeqBatch,
    ARLMLossDict,
    ARLMPredictionDict,
    ARLMModel,
)

__all__ = [
    "RotaryTransformerARLMModel",
    "ARLMLoss",
    "ARLMPredictor",
    "DefaultARLMCollator",
    "ARLMSeq2SeqCollator",
    "ARLMSeq2SeqPredCollator",
    "ARLMBatch",
    "ARLMSeq2SeqBatch",
    "ARLMLossDict",
    "ARLMPredictionDict",
    "ARLMModel",
]
