"""DDiT backbone for Flow Reasoning Models.

The network ``D_θ(x_t, t, c, s)`` maps a continuous interpolant on the
simplex, a scalar time, optional discrete clues, and a self-conditioning
carry to categorical logits.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT

from xlm.model import Model
from xlm.modules.ddit_simple import (
    DDiTLayer,
    DDiTLayerList,
    DDitFinalLayer,
    RotaryEmbedding,
    TimestepEmbedder,
)


class FRMModel(nn.Module, Model):
    """Categorical-prediction DDiT that reads a simplex interpolant."""

    def __init__(
        self,
        num_embeddings: int,
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
        self.num_embeddings = num_embeddings
        self.dim_feedforward = dim_feedforward or 4 * d_model
        self.d_cond = d_cond or d_model // 2
        self.max_length = max_length

        self.vocab_embed = nn.Embedding(
            num_embeddings, d_model, padding_idx=padding_idx
        )
        self.self_cond_proj = nn.Linear(num_embeddings, d_model, bias=False)
        nn.init.zeros_(self.self_cond_proj.weight)
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
        rotary = RotaryEmbedding(
            rotary_emb_dim, head_first=True, cache_size=max_length
        )
        self.encoder = DDiTLayerList.from_layer(
            encoder_layer, num_layers, rotary
        )
        self.output_layer = DDitFinalLayer(
            d_model, num_embeddings, self.d_cond, layer_norm_eps
        )

    def forward(
        self,
        x_t: Float[TT, " *batch seq_len vocab_size"],
        t: Float[TT, " *batch"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        s: Optional[Float[TT, " *batch seq_len vocab_size"]] = None,
        clue_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
        clamp_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
    ) -> Float[TT, " *batch seq_len vocab_size"]:
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)

        # Interpolant / one-hot projected through the token embedding table.
        x = torch.matmul(x_t, self.vocab_embed.weight)

        if clue_ids is not None:
            clue_h = self.vocab_embed(clue_ids)
            if clamp_mask is not None:
                clue_h = clue_h * clamp_mask.to(dtype=clue_h.dtype).unsqueeze(
                    -1
                )
            x = x + clue_h

        if s is not None:
            x = x + self.self_cond_proj(s)

        cond = F.silu(self.sigma_map(t))
        if positions is None:
            if attention_mask is None:
                raise ValueError(
                    "positions is required when attention_mask is None"
                )
            positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)

        for block in self.encoder:
            x = block(x, cond, attention_mask, positions)

        # Zero-init DDiT head starts at 0; add x_t so D can read the interpolant
        # (Stage A ckpt ignored x_t and collapsed to class 8 / digit "1").
        return self.output_layer(x, cond) + x_t

    def get_named_params_for_weight_decay(self):
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                yield (name, param)
