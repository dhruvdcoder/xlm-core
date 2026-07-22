"""LLaDA model: LLaDAModelLM with LLaDA-specific config and MLM-protocol adapter."""

from typing import Optional

from torch import Tensor

from xlm.backbones.llada.modeling_llada import LLaDAModelLM

from .configuration_llada import LLaDAConfig


class LLaDAHFModel(LLaDAModelLM):
    """LLaDA decoder — a pure forward-pass HF model (input_ids -> CausalLMOutputWithPast)."""

    config_class = LLaDAConfig


class LLaDAXLMModel(LLaDAHFModel):
    """Same weights as ``LLaDAHFModel``; ``forward`` matches the MLM predictor protocol.

    Accepts a 2D ``attention_mask`` (handled natively by the LLaDA backbone,
    unlike Dream which needs a 4D expansion), forwards packed ``positions`` as
    RoPE ``position_ids``, and returns logits only (no output dataclass).
    HF checkpoints load without key prefixing. Unlike Dream there is no
    next-token shift: LLaDA predicts each masked position in place, so no
    ``LogitsShiftBy1`` hook is needed.
    """

    def forward(
        self,
        x_t: Tensor,
        attention_mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        output = super().forward(
            input_ids=x_t,
            attention_mask=attention_mask,
            position_ids=positions,
        )
        return output.logits
