# v2
from copy import deepcopy
from sys import argv
from typing import Optional, Protocol, Union
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from xlm.modules.ddit_simple import (
    DDiTLayer,
    DDitFinalLayer,
    RotaryEmbedding,
    TimestepEmbedder,
    DDiTLayerList,
)
from jaxtyping import Integer, Float, Bool
from torch import Tensor as TT
from transformers import PreTrainedTokenizer


class Model(torch.nn.Module):

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        t: Integer[TT, " *batch"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
    ) -> Float[TT, " *batch seq_len vocab_size"]:
        raise NotImplementedError

    def get_named_params_for_weight_decay(self):
        # all parameters except biases and layer-norm parameters
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        # biases and layer-norm parameters
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                yield (name, param)


class MDLMModel(Model):
    """DDiT based transformer that represents time/noise using AdaLN and uses rotary positional embeddings."""

    def __init__(
        self,
        num_embeddings: int,  # vocab plus mask and padding other special tokens
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        d_cond: Optional[int] = None,
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.embed_tokens = nn.Embedding(
            num_embeddings, d_model, padding_idx=padding_idx
        )
        # TODO (neural net): Init embedding with appropriate distribution
        self.d_cond = d_cond or d_model // 2
        self.dim_feedforward = dim_feedforward or 4 * d_model
        self.sigma_map = TimestepEmbedder(self.d_cond, 256)
        encoder_layer = DDiTLayer(
            d_model,
            nhead,
            self.dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            self.d_cond,
            force_flash_attn=force_flash_attn,
        )
        self.max_length = max_length
        self.encoder = DDiTLayerList.from_layer(
            encoder_layer,
            num_layers,
            RotaryEmbedding(
                rotary_emb_dim, head_first=True, cache_size=max_length
            ),
        )
        self.output_layer = DDitFinalLayer(
            d_model, num_embeddings, self.d_cond, layer_norm_eps
        )
        self.num_embeddings = num_embeddings

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        noise: Float[TT, " *batch"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
    ) -> Float[TT, " *batch seq_len vocab_size"]:
        """
        Args:
            x_t: The input tokens of shape (*batch, seq_len)
            noise: The noise of shape (*batch)
            attention_mask: The attention mask of shape (*batch, seq_len), which is True for non-padding tokens.
            positions: The positions of the tokens of shape (*batch, seq_len)
        """
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        x = self.embed_tokens(x_t)  # shape (batch_size, seq_len, d_model)
        # TODO: Why do we need to use silu here?
        c = F.silu(self.sigma_map(noise))
        if positions is None:
            positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
        for block in self.encoder:
            x = block(x, c, attention_mask, positions)

        x = self.output_layer(x, c)  # shape (batch_size, seq_len, vocab_size)
        return x
