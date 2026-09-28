"""Single bidirectional transformer with canvas self-conditioning."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT

from mlm.model_mlm import RotaryTransformerMLMModel


class GemmaUsdmModel(RotaryTransformerMLMModel):
    """Rotary transformer over the concatenated source and canvas.

    Self-conditioning is an additive residual on the token embeddings.
    ``z = 0`` on the source. On the canvas, ``z = FFW(soft)``, where ``soft``
    is a detached ``p_hat E`` from the previous visit. The last linear of
    ``FFW`` is zero-initialized so the residual starts at 0, and it is applied
    inside the visit that should receive its gradient.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.embed_tokens.embedding_dim
        self.self_cond_ffw = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.self_cond_ffw[-1].weight)
        nn.init.zeros_(self.self_cond_ffw[-1].bias)

    def self_conditioning(
        self,
        soft: Float[TT, "batch seq d_model"],
        canvas_mask: Bool[TT, "batch seq"],
    ) -> Float[TT, "batch seq d_model"]:
        """``FFW(soft)`` on the canvas and exact zeros on the source."""
        z = self.self_cond_ffw(soft.to(dtype=self.embed_tokens.weight.dtype))
        gate = canvas_mask.to(dtype=z.dtype).unsqueeze(-1)
        return z * gate

    def forward(
        self,
        x_t: Integer[TT, "batch seq"],
        attention_mask: Optional[Bool[TT, "batch seq"]] = None,
        positions: Optional[Integer[TT, "batch seq"]] = None,
        block_mask=None,
        self_cond: Optional[Float[TT, "batch seq d_model"]] = None,
    ) -> Float[TT, "batch seq vocab"]:
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        x = self.embed_tokens(x_t)
        if self_cond is not None:
            x = x + self_cond.to(dtype=x.dtype)

        if block_mask is not None:
            attention_mask = None

        for block in self.encoder:
            x = block(
                x,
                attention_mask,
                positions=positions,
                block_mask=block_mask,
            )
        return self.output_layer(x)
