"""
Idlm - Core Language Model for XLM Framework

This module implements the Idlm model with all necessary components:
- Model architecture (model.py)
- Loss function (loss.py)
- Predictor for inference (predictor.py)
- Data collators (collators.py)
- Metrics computation (metrics.py)
- Type definitions (types.py)
- Noise schedules (noise_schedule.py)

This is a core model that is part of the XLM library.
"""

from .model import IdlmModel, DDITIDLMModel
from .loss import IdlmLoss
from .predictor import IdlmPredictor
from .collators import (
    DefaultIdlmCollator,
    IdlmSeq2SeqCollator,
    IdlmSeq2SeqPredCollator,
)
from .noise_schedule import (
    PoissonNoiseSchedule,
    LogLinearNoiseSchedule,
    GeometricNoiseSchedule,
)
from .datamodule import print_batch_idlm
from .nn import incomplete_gamma_factor_using_series
from .types import (
    IdlmBatch,
    IdlmSeq2SeqBatch,
    IdlmLossDict,
    IdlmPredictionDict,
    TokenLogitsType,
    LengthLogitsType,
)

__all__ = [
    "IdlmModel",
    "DDITIDLMModel",  # Keep for backward compatibility
    "IdlmLoss",
    "IdlmPredictor",
    "DefaultIdlmCollator",
    "IdlmSeq2SeqCollator",
    "IdlmSeq2SeqPredCollator",
    "print_batch_idlm",
    "PoissonNoiseSchedule",
    "LogLinearNoiseSchedule",
    "GeometricNoiseSchedule",
    "incomplete_gamma_factor_using_series",
    "IdlmBatch",
    "IdlmSeq2SeqBatch",
    "IdlmLossDict",
    "IdlmPredictionDict",
    "TokenLogitsType",
    "LengthLogitsType",
]
