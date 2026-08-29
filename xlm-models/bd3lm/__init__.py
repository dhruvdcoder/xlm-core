"""BD3-LM - Block Discrete Denoising Diffusion Language Model.

Autoregressive across blocks, diffusion within a block.

    xlm job_type=train job_name=my_run experiment=star_medium_bd3lm   # seq2seq
    xlm job_type=train job_name=my_run experiment=owt_bd3lm           # unconditional

See README.md for the experiments, model sizes and fine-tuning options.
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
from .predictor_bd3lm import Bd3lmPredictor, Bd3lmUnconditionalPredictor
from .datamodule_bd3lm import (
    DefaultBd3lmCollator,
    Bd3lmSeq2SeqCollator,
    Bd3lmSeq2SeqPredCollator,
    Bd3lmUnconditionalPredCollator,
    Bd3lmEmptyDataset,
)
from .noise_schedule import Bd3lmNoise


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
