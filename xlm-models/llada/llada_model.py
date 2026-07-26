"""LLaDA model: LLaDAModelLM with LLaDA-specific config and MLM-protocol adapter."""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

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
        h_t: Optional[Tensor] = None,
    ) -> Tensor:
        del h_t  # vanilla path ignores carry
        output = super().forward(
            input_ids=x_t,
            attention_mask=attention_mask,
            position_ids=positions,
        )
        return output.logits

    @property
    def d_model(self) -> int:
        return int(self.config.d_model)


class LLaDARelayModel(LLaDAXLMModel):
    """LLaDA adapter with optional RELAY carry inject/readout.

    When ``use_relay=True``, ``forward`` returns ``(logits, h_s)``. ``h_t`` is
    LayerNorm'd and added at mask-token positions only (zero-init LN weight so
    step 0 is a near-identity). ``h_s`` is the hidden state after the selected
    transformer block (before ``ln_f``).
    """

    def __init__(
        self,
        config: LLaDAConfig,
        model=None,
        init_params: bool = False,
        use_relay: bool = False,
        relay_layer: int = -1,
        **kwargs,
    ):
        super().__init__(config=config, model=model, init_params=init_params, **kwargs)
        self.use_relay = bool(use_relay)
        n_layers = int(self.config.n_layers)
        resolved = int(relay_layer) % n_layers
        self.relay_layer = resolved
        if self.use_relay:
            self.relay_layer_norm = nn.LayerNorm(
                int(self.config.d_model), eps=float(self.config.rms_norm_eps)
            )
            nn.init.zeros_(self.relay_layer_norm.weight)

    def forward(
        self,
        x_t: Tensor,
        attention_mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
        h_t: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if not self.use_relay:
            return super().forward(x_t, attention_mask, positions)

        # Inject via wte forward-hook so embedding stays on the normal FSDP
        # forward path (a direct wte(...) call under use_orig_params=False can
        # hit a sharded/empty weight and device-side index asserts).
        captured: dict = {}
        handles = []

        if h_t is not None:
            mask_id = int(self.config.mask_token_id)

            def _embed_hook(_module, inp, out):
                ids = inp[0]
                delta = self.relay_layer_norm(h_t.to(dtype=out.dtype))
                guard = (ids == mask_id).unsqueeze(-1)
                return torch.where(guard, out + delta, out)

            handles.append(
                self.model.transformer.wte.register_forward_hook(_embed_hook)
            )

        def _hs_hook(_module, _inp, out):
            # LLaDALlamaBlock returns (hidden, cache)
            captured["h_s"] = out[0] if isinstance(out, tuple) else out

        handles.append(
            self.model.transformer.blocks[self.relay_layer].register_forward_hook(
                _hs_hook
            )
        )
        try:
            output = LLaDAHFModel.forward(
                self,
                input_ids=x_t,
                attention_mask=attention_mask,
                position_ids=positions,
            )
        finally:
            for handle in handles:
                handle.remove()

        h_s = captured.get("h_s")
        if h_s is None:
            raise RuntimeError(
                f"Failed to capture h_s from relay_layer={self.relay_layer}"
            )
        return output.logits, h_s
