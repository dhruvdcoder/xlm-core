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
from xlm.model import Model
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

########################################################
# region: ARLM Transformer


class RotaryTransformerARLMModel(torch.nn.Module, Model):
    """Rotary embedding based transformer decoder for auto-regressive language modeling."""

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
        final_layer_without_normalization: bool = False,
    ):
        """Initialize the ARLM transformer model.

        Args:
            num_embeddings: Size of the vocabulary.
            d_model: Dimension of the model.
            num_layers: Number of transformer layers.
            nhead: Number of attention heads.
            padding_idx: Index of the padding token.
            dim_feedforward: Dimension of the feedforward network.
            dropout: Dropout rate.
            activation: Activation function.
            layer_norm_eps: Epsilon for layer normalization.
            rotary_emb_dim: Dimension of rotary embeddings.
            max_length: Maximum sequence length.
            force_flash_attn: Whether to force flash attention.
            final_layer_without_normalization: Whether to use final layer without normalization.
        """
        super().__init__()
        self.padding_idx = padding_idx
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
            zero_init=False,
        )

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        attention_mask: Optional[Bool[TT, " *batch seq_len seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        token_type_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
    ) -> Float[TT, " *batch seq_len vocab_size"]:
        """
        Forward pass of the ARLM model.

        Args:
            x_t: The input tokens of shape (*batch, seq_len)
            attention_mask: The attention mask of shape (*batch, seq_len, seq_len) for full attention matrix,
                          or (*batch, seq_len) for simple mask. True for non-padding tokens.
            positions: The positions of the tokens of shape (*batch, seq_len)
            token_type_ids: The token type ids of shape (*batch, seq_len)

        Returns:
            vocab_logits: The vocabulary logits of shape (*batch, seq_len, vocab_size)
        """
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        x = self.embed_tokens(x_t)  # shape (batch_size, seq_len, d_model)

        for block in self.encoder:
            x = block(x, attention_mask, positions=positions)

        vocab_logits = self.output_layer(
            x,
        )  # shape (batch_size, seq_len, vocab_size)
        return vocab_logits

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


# endregion: ARLM Transformer
########################################################
