"""
ILM - Infilling Language Model for xLM Framework

This package implements the ILM model with all necessary components:
- Model architecture (model_ilm.py)
- Loss function (loss_ilm.py)
- Predictor for inference (predictor_ilm.py)
- Data module (datamodule_ilm.py)
- Metrics computation (metrics_ilm.py)
- Type definitions (types_ilm.py)
- Neural network utilities (nn.py)

This model was migrated from xlm.lm.ilm to be an external model.
"""

from .model_ilm import (
    RotaryTransformerILMModel,
    RotaryTransformerILMModelWithClassification,
    RotaryTransformerILMModelWithStoppingClassification,
    RotaryTransformerILMModelWithLengthClassification,
    GPT2ILMModel,
    GPT2ILMModelWithClassification,
    GPT2ILMModelWithStoppingClassification,
    GPT2ILMModelWithLengthClassification,
)
from .loss_ilm import ILMLossWithMaskedCE
from .predictor_ilm import (
    ILMPredictor,
    ILMPredictorWithLengthClassification,
    ILMPredictorWithStoppingClassification,
)
from .datamodule_ilm import (
    DefaultILMCollator,
    ILMSeq2SeqCollator,
    ILMSeq2SeqPredCollator,
)
from .types_ilm import (
    ILMBatch,
    ILMSeq2SeqPredictionBatch,
    ILMUncondtionalPredictionBatch,
    ILMInfillPredictionBatch,
    ILMLossDict,
    ILMModel,
    ILMPredictionDict,
)

__all__ = [
    "RotaryTransformerILMModel",
    "RotaryTransformerILMModelWithClassification",
    "RotaryTransformerILMModelWithStoppingClassification",
    "RotaryTransformerILMModelWithLengthClassification",
    "GPT2ILMModel",
    "GPT2ILMModelWithClassification",
    "GPT2ILMModelWithStoppingClassification",
    "GPT2ILMModelWithLengthClassification",
    "ILMLossWithMaskedCE",
    "ILMPredictor",
    "ILMPredictorWithLengthClassification",
    "ILMPredictorWithStoppingClassification",
    "DefaultILMCollator",
    "ILMSeq2SeqCollator",
    "ILMSeq2SeqPredCollator",
    "ILMBatch",
    "ILMSeq2SeqPredictionBatch",
    "ILMUncondtionalPredictionBatch",
    "ILMInfillPredictionBatch",
    "ILMLossDict",
    "ILMModel",
    "ILMPredictionDict",
]
