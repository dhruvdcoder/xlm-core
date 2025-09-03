from typing import Any, Dict, List, Optional, Tuple, cast, Literal, Callable
from itertools import cycle
from functools import partial
import torch
from jaxtyping import Bool, Integer
from xlm import flags
from xlm.datamodule import Tokenizer
from torch import Tensor as TT
from xlm.harness import Predictor
from xlm.noise import NoiseSchedule
from .types_indigo import IndigoBatch, IndigoPredictionDict

import time


###############################################################
# region: Predictors


class IndigoPredictor(
    torch.nn.Module,
    Predictor[IndigoBatch, IndigoPredictionDict],
):

    def __init__(
        self,
        max_steps: int,
        max_length: int,
        tokenizer: Optional[Tokenizer] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
    ):
        """Constructor for IndigoPredictor."""
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        super().__init__()
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_length = max_length
        self.noise_schedule = noise_schedule

    @torch._dynamo.disable()
    def predict(
        self,
        batch: Dict[str, Any],  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        # TODO (URV): Implement the predictor.
        pass


# endregion: Predictors
###############################################################
