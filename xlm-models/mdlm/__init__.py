"""
MDLM - Masked Diffusion Language Model for XLM Framework

This package implements the MDLM model with all necessary components:
- Model architecture (model_mdlm.py)
- Loss function (loss_mdlm.py)
- Predictor for inference (predictor_mdlm.py)
- Data module (datamodule_mdlm.py)
- Metrics computation (metrics_mdlm.py)
- Type definitions (types_mdlm.py)
- Noise functions (noise_mdlm.py)

This model was migrated from xlm.lm.mdlm to be an external model.
"""

from .model_mdlm import MDLMModel
from .loss_mdlm import MDLMLoss
from .predictor_mdlm import MDLMPredictor
from .datamodule_mdlm import (
    DefaultMDLMCollator,
    MDLMSeq2SeqTrainCollator,
    MDLMSeq2SeqPredCollator,
)
from .types_mdlm import (
    MDLMBatch,
    MDLMSeq2SeqPredictionBatch,
    MDLMUncondtionalPredictionBatch,
    MDLMLossDict,
    MDLMModel,
    MDLMPredictionDict,
)

__all__ = [
    "MDLMModel",
    "MDLMLoss",
    "MDLMPredictor",
    "DefaultMDLMCollator",
    "MDLMSeq2SeqTrainCollator",
    "MDLMSeq2SeqPredCollator",
    "MDLMBatch",
    "MDLMSeq2SeqPredictionBatch",
    "MDLMUncondtionalPredictionBatch",
    "MDLMLossDict",
    "MDLMModel",
    "MDLMPredictionDict",
]
