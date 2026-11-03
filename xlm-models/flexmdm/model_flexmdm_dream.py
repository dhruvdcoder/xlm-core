"""Dream-Coder FlexMDM: Dream backbone + AdaLN time conditioning + insertion head.

Port of ``genuine-any-order/FlexMDM/flexmdm/architecture.py`` onto xlm-core's
vendored :class:`xlm.backbones.dream.modeling_dream.DreamModelCore`.

Differences from the reference wrapper (all weight-compatible):

- The reference applies AdaLN through ``register_forward_hook`` on each layer's
  ``input_layernorm`` / ``post_attention_layernorm`` with one ``temb_mods[i]``
  Linear per layer. Here the same ``Linear(hidden, 2*hidden)`` lives inside each
  :class:`DreamDecoderLayer` as ``rms_norm_modulations`` (config
  ``time_conditioning="flexmdm_shared2"``) and the backbone receives the
  conditioning vector through ``forward(c=...)``. :meth:`remap_extras_state_dict`
  renames ``temb_mods.{i}.*`` -> ``model.layers.{i}.rms_norm_modulations.modulation.*``.
- ``forward(x, t, attention_mask, positions)`` returns ``(token_logits, log_rate)``
  (the FlexMDM ``(logits, length)`` protocol used by the loss and predictor),
  instead of a dict.

State-dict layout (checkpoint ``yuyuanchen0/flexmdm``):

- Dream HF safetensors: ``model.*``, ``lm_head.weight`` (loaded via ``hub.repo_id``).
- ``flexmdm_extras.pt``: ``insertion_head.*``, ``time_mlp.*``, ``temb_mods.*``
  (loaded via ``hub.extra_state_dict_files`` after remapping).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT

from xlm.backbones.dream.modeling_dream import DreamModelCore
from xlm.utils.rank_zero import RankedLogger

from .configuration_flexmdm_dream import FlexMDMDreamConfig

logger = RankedLogger(__name__, rank_zero_only=True)

EXTRAS_PREFIX_TEMB_MODS = "temb_mods."
EXTRAS_PREFIXES = ("insertion_head.", "time_mlp.", EXTRAS_PREFIX_TEMB_MODS)
# Parameter-name fragments that belong to the FlexMDM additions (used for the
# insertion LR multiplier; mirrors ``train_fsdp.INSERTION_MODULE_FRAGMENTS``).
INSERTION_PARAM_FRAGMENTS = ("insertion_head", "time_mlp", "rms_norm_modulations")


def _linear_default_reset_(linear: nn.Linear) -> None:
    """``nn.Linear.reset_parameters`` using tensor ops only.

    ``transformers.modeling_utils.no_init_weights`` (used by
    ``skip_init_weights=true``) monkeypatches ``torch.nn.init.*`` into no-ops,
    which leaves freshly constructed Linear layers uninitialized. The FlexMDM
    additions are not in the pretrained Dream checkpoint, so re-run the default
    init with ``Tensor`` methods (which are not patched).
    """
    fan_in = linear.weight.shape[1]
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    # kaiming_uniform_(a=sqrt(5)) == U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    linear.weight.data.uniform_(-bound, bound)
    if linear.bias is not None:
        linear.bias.data.uniform_(-bound, bound)


class InsertionHead(nn.Module):
    """Predict the log of the per-position insertion rate (FlexMDM reference head).

    ``LayerNorm -> Linear(H, H) -> GELU -> Linear(H, 1)``; the scalar output is
    clamped to ``[-15, 15]`` so ``exp(z)`` is representable in bf16. The rate is
    ``exp(z)`` and is computed by the loss / sampler.
    """

    Z_CLAMP_MIN = -15.0
    Z_CLAMP_MAX = 15.0

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.weight.data.fill_(1.0)
        self.norm.bias.data.zero_()
        _linear_default_reset_(self.proj1)
        self.proj2.weight.data.normal_(mean=0.0, std=1e-3)
        self.proj2.bias.data.fill_(-1.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        x = self.act(self.proj1(x))
        x = self.proj2(x).squeeze(-1)
        return x.clamp(min=self.Z_CLAMP_MIN, max=self.Z_CLAMP_MAX)


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    *,
    time_scale: float = 1000.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """FlexMDM reference ``FlexMDM.timestep_embedding`` (float32, shape ``(B, dim)``)."""
    if timesteps.ndim != 1:
        raise ValueError(
            f"timesteps must have shape (batch,), got {tuple(timesteps.shape)}."
        )
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(
            start=0, end=half, dtype=torch.float32, device=timesteps.device
        )
        / half
    )
    args = (timesteps.float() * time_scale)[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class FlexMDMDreamModel(DreamModelCore):
    """Dream-Coder FlexMDM: ``forward(x, t, attention_mask, positions) -> (logits, log_rate)``."""

    config_class = FlexMDMDreamConfig

    def __init__(
        self,
        config: FlexMDMDreamConfig,
        max_length: int = 768,
        insertion_lr_multiplier: float = 1.0,
        base_lr: Optional[float] = None,
        force_flash_attn: bool = False,
    ):
        if not config.use_time_embedding or config.time_conditioning != "flexmdm_shared2":
            raise ValueError(
                "FlexMDMDreamModel requires config.use_time_embedding=True and "
                "config.time_conditioning='flexmdm_shared2'."
            )
        if int(config.d_cond) != int(config.hidden_size):
            raise ValueError(
                "FlexMDMDreamModel requires d_cond == hidden_size "
                f"(got d_cond={config.d_cond}, hidden_size={config.hidden_size})."
            )
        super().__init__(config)
        hidden = int(config.hidden_size)
        self.hidden_size = hidden
        self.max_length = int(max_length)
        self.time_scale = float(getattr(config, "time_scale", 1000.0))
        self.time_max_period = int(getattr(config, "time_max_period", 10000))
        self.insertion_lr_multiplier = float(insertion_lr_multiplier)
        self.base_lr = base_lr
        self.force_flash_attn = force_flash_attn

        self.insertion_head = InsertionHead(hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.reset_flexmdm_parameters()

    # ------------------------------------------------------------------
    # Initialization / weight loading
    # ------------------------------------------------------------------
    def reset_flexmdm_parameters(self) -> None:
        """Initialize the FlexMDM additions exactly like the reference wrapper.

        - ``time_mlp``: PyTorch default Linear init.
        - ``insertion_head``: default init, ``proj2.weight ~ N(0, 1e-3)``, ``proj2.bias = -1``.
        - per-layer modulations (``temb_mods``): zeros (identity AdaLN).

        Safe to call under ``no_init_weights`` (uses tensor ops only).
        """
        for module in self.time_mlp:
            if isinstance(module, nn.Linear):
                _linear_default_reset_(module)
        self.insertion_head.reset_parameters()
        for layer in self.model.layers:
            mod = layer.rms_norm_modulations.modulation
            mod.weight.data.zero_()
            if mod.bias is not None:
                mod.bias.data.zero_()

    def init_new_parameters(self, tokenizer: Any = None) -> None:  # noqa: ARG002
        """Hook called by ``FSDPHarness.on_fit_start`` at global step 0.

        The FlexMDM additions are already initialized in ``__init__`` (so the
        init also happens on the CPU copy before FSDP wrapping / weight loading
        order issues); nothing to do here.
        """
        return None

    @staticmethod
    def remap_extras_state_dict(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Rename reference ``flexmdm_extras.pt`` keys to this module's layout."""
        remapped: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(EXTRAS_PREFIX_TEMB_MODS):
                # temb_mods.{i}.{weight,bias} -> model.layers.{i}.rms_norm_modulations.modulation.{weight,bias}
                rest = key[len(EXTRAS_PREFIX_TEMB_MODS) :]
                layer_idx, param = rest.split(".", 1)
                remapped[
                    f"model.layers.{layer_idx}.rms_norm_modulations.modulation.{param}"
                ] = value
            else:
                remapped[key] = value
        return remapped

    @staticmethod
    def to_extras_state_dict(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Inverse of :meth:`remap_extras_state_dict`: extract reference-layout extras."""
        extras: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(("insertion_head.", "time_mlp.")):
                extras[key] = value
            elif ".rms_norm_modulations.modulation." in key and key.startswith(
                "model.layers."
            ):
                layer_idx = key.split(".")[2]
                param = key.rsplit(".", 1)[1]
                extras[f"temb_mods.{layer_idx}.{param}"] = value
        return extras

    def load_extra_state_dict(
        self, state_dict: Dict[str, torch.Tensor], *, strict: bool = True
    ):
        """Load a reference ``flexmdm_extras.pt`` state dict (keys remapped)."""
        remapped = self.remap_extras_state_dict(state_dict)
        result = self.load_state_dict(remapped, strict=False)
        unexpected = list(result.unexpected_keys)
        own_extras = {
            k
            for k in self.state_dict().keys()
            if k.startswith(("insertion_head.", "time_mlp."))
            or ".rms_norm_modulations." in k
        }
        missing = sorted(own_extras - set(remapped.keys()))
        if strict and (unexpected or missing):
            raise RuntimeError(
                "FlexMDM extras load failed: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if unexpected or missing:
            logger.warning(
                f"FlexMDM extras: missing={missing}, unexpected={unexpected}"
            )
        logger.info(f"FlexMDM extras applied: {len(remapped)} tensors.")
        return result

    # ------------------------------------------------------------------
    # Optimizer groups (insertion LR multiplier)
    # ------------------------------------------------------------------
    def flexmdm_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        for name, param in self.named_parameters():
            if any(frag in name for frag in INSERTION_PARAM_FRAGMENTS):
                yield name, param

    def get_param_groups(self) -> List[Dict[str, Any]]:
        """Two groups: backbone (base lr) and FlexMDM additions (``base_lr * multiplier``).

        Mirrors ``train_fsdp.build_optimizer``: weight decay applies to both groups
        (the reference does not exclude biases / norms).
        """
        backbone_params: List[nn.Parameter] = []
        insertion_params: List[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(frag in name for frag in INSERTION_PARAM_FRAGMENTS):
                insertion_params.append(param)
            else:
                backbone_params.append(param)
        groups: List[Dict[str, Any]] = [{"params": backbone_params}]
        if insertion_params:
            group: Dict[str, Any] = {"params": insertion_params}
            if self.insertion_lr_multiplier != 1.0:
                if self.base_lr is None:
                    raise ValueError(
                        "insertion_lr_multiplier != 1 requires base_lr "
                        "(e.g. model.base_lr: ${optimizer.lr})."
                    )
                group["lr"] = float(self.base_lr) * self.insertion_lr_multiplier
            groups.append(group)
        logger.info(
            "FlexMDMDreamModel param groups: backbone="
            f"{sum(p.numel() for p in backbone_params):,}, insertion="
            f"{sum(p.numel() for p in insertion_params):,}"
            + (
                f" (lr={groups[1]['lr']})"
                if len(groups) > 1 and "lr" in groups[1]
                else ""
            )
        )
        return groups

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def time_conditioning(self, t: torch.Tensor) -> torch.Tensor:
        """``time_mlp(sinusoidal(t))`` in the ``time_mlp`` dtype, shape ``(B, hidden)``."""
        emb = sinusoidal_timestep_embedding(
            t,
            self.hidden_size,
            time_scale=self.time_scale,
            max_period=self.time_max_period,
        )
        weight = self.time_mlp[0].weight
        emb = emb.to(device=weight.device, dtype=weight.dtype)
        return self.time_mlp(emb)

    @staticmethod
    def _to_attn_mask_4d(attention_mask: Optional[TT]) -> Optional[TT]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            return attention_mask
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        # Reference: ``attention_mask.bool()[:, None, None, :]`` (key padding only).
        return attention_mask[:, None, None, :]

    def forward(  # type: ignore[override]
        self,
        x: Integer[TT, "batch seq_len"],
        t: Float[TT, "batch"],
        attention_mask: Optional[Bool[TT, "batch seq_len"]] = None,
        positions: Optional[Integer[TT, "batch seq_len"]] = None,
    ) -> Tuple[Float[TT, "batch seq_len vocab"], Float[TT, "batch seq_len"]]:
        """Return ``(token_logits, log_rate)``.

        ``token_logits[:, i]`` follow the Dream convention (prediction for position
        ``i + 1``); the loss / predictor apply the right shift. ``log_rate[:, i]`` is
        the log expected number of insertions *after* token ``i``.
        """
        if t is None:
            raise ValueError("FlexMDMDreamModel.forward requires `t` (shape (batch,)).")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(
                f"t must have shape (batch,)=({x.shape[0]},), got {tuple(t.shape)}."
            )
        c = self.time_conditioning(t.to(x.device))
        attn_4d = (
            None if self.force_flash_attn else self._to_attn_mask_4d(attention_mask)
        )
        outputs = self.model(
            input_ids=x,
            attention_mask=attn_4d,
            position_ids=positions,
            return_dict=True,
            use_cache=False,
            c=c,
        )
        hidden_states = outputs.last_hidden_state
        token_logits = self.lm_head(hidden_states)
        log_rate = self.insertion_head(hidden_states)
        return token_logits, log_rate


__all__ = [
    "FlexMDMDreamModel",
    "FlexMDMDreamConfig",
    "InsertionHead",
    "sinusoidal_timestep_embedding",
    "INSERTION_PARAM_FRAGMENTS",
    "EXTRAS_PREFIXES",
]
