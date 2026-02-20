"""
IDLM - Iterative Diffusion Language Model for xLM Framework

This package implements the IDLM model with all necessary components:
- Model architecture (model_idlm.py)
- Loss function (loss_idlm.py)
- Predictor for inference (predictor_idlm.py)
- Data module (datamodule_idlm.py)
- Metrics computation (metrics_idlm.py)
- Type definitions (types_idlm.py)
- Noise schedules (noise_schedule.py)
- Neural network utilities (nn.py)

This model was migrated from xlm.lm.idlm to be an external model.
"""

from .model_idlm import DDITIDLMModel, DDITIDLMModelFinalLength
from .loss_idlm import IdlmLoss
from .predictor_idlm import IdlmPredictor
from .datamodule_idlm import (
    DefaultIdlmCollator,
    IdlmSeq2SeqCollator,
    IdlmSeq2SeqPredCollator,
)
from .types_idlm import (
    IdlmBatch,
    IdlmSeq2SeqBatch,
    IdlmLossDict,
    IdlmPredictionDict,
    IdlmModel,
)
from .noise_schedule import (
    PoissonNoiseSchedule,
    LogLinearNoiseSchedule,
    GeometricNoiseSchedule,
)

__all__ = [
    "DDITIDLMModel",
    "DDITIDLMModelFinalLength",
    "IdlmLoss",
    "IdlmPredictor",
    "DefaultIdlmCollator",
    "IdlmSeq2SeqCollator",
    "IdlmSeq2SeqPredCollator",
    "IdlmBatch",
    "IdlmSeq2SeqBatch",
    "IdlmLossDict",
    "IdlmPredictionDict",
    "IdlmModel",
    "PoissonNoiseSchedule",
    "LogLinearNoiseSchedule",
    "GeometricNoiseSchedule",
]
