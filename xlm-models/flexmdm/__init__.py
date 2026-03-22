"""
FlexMDM - Flexible Masked Diffusion Model for XLM Framework

This package implements the FlexMDM model with all necessary components:
- Model architecture (model_flexmdm.py)
- Loss function (loss_flexmdm.py)
- Predictor for inference (predictor_flexmdm.py)
- Data module (datamodule_flexmdm.py)
- Metrics computation (metrics_flexmdm.py)
- Type definitions (types_flexmdm.py)
- Noise schedules (noise_flexmdm.py)
"""

from .model_flexmdm import FlexMDMModel
from .loss_flexmdm import FlexMDMLoss
from .predictor_flexmdm import FlexMDMPredictor
from .datamodule_flexmdm import (
    DefaultFlexMDMCollator,
    DefaultFlexMDMUnconditionalPredCollator,
    FlexMDMSeq2SeqTrainCollator,
    FlexMDMSeq2SeqPredCollator,
)
from .types_flexmdm import (
    FlexMDMBatch,
    FlexMDMSeq2SeqPredictionBatch,
    FlexMDMUncondtionalPredictionBatch,
    FlexMDMLossDict,
    FlexMDMPredictionDict,
)
from .noise_flexmdm import FlexMDMNoiseSchedule

__all__ = [
    "FlexMDMModel",
    "FlexMDMLoss",
    "FlexMDMPredictor",
    "DefaultFlexMDMCollator",
    "DefaultFlexMDMUnconditionalPredCollator",
    "FlexMDMSeq2SeqTrainCollator",
    "FlexMDMSeq2SeqPredCollator",
    "FlexMDMBatch",
    "FlexMDMSeq2SeqPredictionBatch",
    "FlexMDMUncondtionalPredictionBatch",
    "FlexMDMLossDict",
    "FlexMDMPredictionDict",
    "FlexMDMNoiseSchedule",
]
