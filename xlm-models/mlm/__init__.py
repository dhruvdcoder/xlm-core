"""
MLM - Masked Language Model for xLM Framework

This package implements the MLM model with all necessary components:
- Model architecture (model_mlm.py)
- Loss function (loss_mlm.py)
- Predictor for inference (predictor_mlm.py)
- Data module (datamodule_mlm.py)
- Metrics computation (metrics_mlm.py)
- Type definitions (types_mlm.py)
- History tracking (history_mlm.py)

This model was migrated from xlm.lm.mlm to be an external model.
"""

from .model_mlm import RotaryTransformerMLMModel
from .loss_mlm import MLMLoss
from .predictor_mlm import MLMPredictor
from .datamodule_mlm import (
    DefaultMLMCollator,
    MLMSeq2SeqTrainCollator,
    MLMSeq2SeqCollator,
    MLMSeq2SeqPredCollator,
)
from .types_mlm import (
    MLMBatch,
    MLMSeq2SeqPredictionBatch,
    MLMUncondtionalPredictionBatch,
    MLMLossDict,
    MLMModel,
    MLMPredictionDict,
)
from .history_mlm import HistoryTopKPlugin

__all__ = [
    "RotaryTransformerMLMModel",
    "MLMLoss",
    "MLMPredictor",
    "DefaultMLMCollator",
    "MLMSeq2SeqTrainCollator",
    "MLMSeq2SeqCollator",
    "MLMSeq2SeqPredCollator",
    "MLMBatch",
    "MLMSeq2SeqPredictionBatch",
    "MLMUncondtionalPredictionBatch",
    "MLMLossDict",
    "MLMModel",
    "MLMPredictionDict",
    "HistoryTopKPlugin",
]
