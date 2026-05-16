# v2
from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT

from xlm.modules.rotary_transformer import (
    RotaryTransformerFinalLayer,
    RotaryTransformerLayer,
    RotaryTransformerLayerList,
    RotaryEmbedding,
)

from xlm.utils.rank_zero import RankedLogger
from xlm.model import Model
logger = RankedLogger(__name__, rank_zero_only=True)

########################################################
# region: Rotary Transformer


class RotaryTransformerMLMModel(torch.nn.Module, Model):
    "Rotary embedding based transformer decoder."

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
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
        use_flex_attn: bool = False,
        final_layer_without_normalization: bool = False,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.use_flex_attn = use_flex_attn
        self.embed_tokens = nn.Embedding(
            num_embeddings, d_model, padding_idx=padding_idx
        )
        self.dim_feedforward = dim_feedforward or 4 * d_model
        encoder_layer = RotaryTransformerLayer(
            d_model,
            nhead,
            self.dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            force_flash_attn=force_flash_attn,
        )
        self.max_length = max_length
        self.encoder = RotaryTransformerLayerList.from_layer(
            encoder_layer,
            num_layers,
            RotaryEmbedding(
                rotary_emb_dim, head_first=True, cache_size=max_length
            ),
        )
        self.output_layer = RotaryTransformerFinalLayer(
            d_model,
            num_embeddings,
            layer_norm_eps,
            use_final_layer_norm=not final_layer_without_normalization,
            zero_init=False,  # zero init important for mdlm, mlm?
        )

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        block_mask=None,
    ) -> Float[TT, " *batch seq_len vocab_size"]:
        """
        Args:
            x_t: The input tokens of shape (*batch, seq_len)
            attention_mask: Boolean mask ``(*batch, seq_len)`` — valid (non-padding)
                tokens.  Ignored when ``block_mask`` is set (packed + FlexAttention).
            positions: Per-token RoPE positions ``(*batch, seq_len)``.  Required for
                packed sequences (reset at each segment boundary).
            block_mask: FlexAttention ``BlockMask`` for packed training when
                ``use_flex_attn=True``.  When set, ``attention_mask`` is ignored.
        """
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        x = self.embed_tokens(x_t)  # shape (batch_size, seq_len, d_model)

        if block_mask is not None:
            # BlockMask from PackedMLMCollator; moved to device in MLMLoss.__call__.
            attention_mask = None

        for block in self.encoder:
            x = block(x, attention_mask, positions=positions, block_mask=block_mask)

        vocab_logits = self.output_layer(
            x,
        )  # shape (batch_size, seq_len, vocab_size)
        return vocab_logits

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
