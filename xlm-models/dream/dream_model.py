"""Dream model: DreamModelCore with Dream-specific config and MLM-protocol adapter."""

from typing import Optional

import torch
from torch import Tensor

from xlm.backbones.dream.modeling_dream import DreamModelCore
from .configuration_dream import DreamConfig


class DreamModel(DreamModelCore):
    """Dream decoder — a pure forward-pass model (input_ids -> logits)."""

    config_class = DreamConfig


class DreamBackbone(DreamModel):
    """Same weights as ``DreamModel``; ``forward`` matches the MLM predictor protocol.

    Accepts 2D ``attention_mask``, converts to 4D for Dream attention, and returns
    logits only (no ``MaskedLMOutput``). HF checkpoints load without key prefixing.
    """

    def forward(
        self,
        x_t: Tensor,
        attention_mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        if attention_mask is not None:
            attn_mask_4d = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            attn_mask_4d = None
        output = super().forward(
            input_ids=x_t,
            attention_mask=attn_mask_4d,
            position_ids=positions,
        )
        return output.logits
