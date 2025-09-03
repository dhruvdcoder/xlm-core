from typing import Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT

from xlm.modules.rotary_transformer import (
    RotaryTransformerFinalLayer,
    RotaryTransformerFinalLayerForClassification,
    RotaryTransformerLayer,
    RotaryTransformerLayerList,
    RotaryEmbedding,
)
from xlm.modules.gpt2_transformer import GPT, GPTConfig

from xlm.model import Model
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


# TODO (URV): Implement a new transformer model that uses tertiary positional embeddings like shown in the indigo paper.
# you can use the rotary transformer for reference but the way you encode the tertiary positional embeddings will be different.


########################################################
# region: Indigo Transformer

# endregion: Indigo Transformer
########################################################
