from typing import Optional, Tuple, Dict, Any

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
from .types_indigo import IndigoBatch

logger = RankedLogger(__name__, rank_zero_only=True)


# TODO (URV): Implement a new transformer model that uses tertiary positional embeddings like shown in the indigo paper.
# you can use the rotary transformer for reference but the way you encode the tertiary positional embeddings will be different.


########################################################
# region: Indigo Transformer


class IndigoModel(nn.Module):
    """Indigo model implementation."""

    def __init__( 
            self,
            num_embeddings: int,  # vocab plus padding and other special tokens
            d_model: int,
            num_layers: int,
            nhead: int,
            padding_idx: int = 0,
            dim_feedforward: Optional[int] = None,
            dropout: float = 0.1,
            activation: str = "relu",
            layer_norm_eps: float = 1e-5,
            rotary_emb_dim: int = 64,
            max_length: int = 1024,
            force_flash_attn: bool = False,
            final_layer_without_normalization: bool = False
        ):

        super().__init__()
        # TODO (URV): Initialize your model components
       
       # Initialize other things among the two layers below
       # Get list of IndigoTransformerLayer from IndigoTransformerLayerList here
       # Get the IndigoTransformerFinalLayer
       # Get the IndigoTransformerPointerNetwork layer initialized 
       # get_position would use the IndigoTransformerPointerNetwork, it wont be used in forward
       # IndigoTransformerLayer, IndigoTransformerLayerList, IndigoTransformerFinalLayer are used in forward
       
        

    def forward(
        self,
        input_ids: Integer[TT, " batch seq_len"],
        attention_mask: Optional[Integer[TT, " batch seq_len"]] = None,
        **kwargs,
    ):
        # TODO (URV): Implement your forward pass
        pass
    
    def get_position(self):
        pass

    def get_named_params_for_weight_decay(self):
        """Get parameters for weight decay (all parameters except biases and layer-norm parameters)."""
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        """Get parameters for no weight decay (biases and layer-norm parameters)."""
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                yield (name, param)


# endregion: Indigo Transformer
########################################################

class IndigoTransformerLayer(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

class IndigoTransformerLayerList(nn.ModuleList):
    def __init__(self):
        pass

    @classmethod
    def from_layer(self):
        pass

class IndigoTransformerFinalLayer(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

class IndigoTransformerPointerNetwork(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass
