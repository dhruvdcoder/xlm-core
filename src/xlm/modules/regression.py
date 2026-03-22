from __future__ import annotations

import torch
from typing import Literal, Optional
from torch import nn
from xlm.modules.ddit_simple_v2 import AdaLNModulations, LayerNormAndScale


class RateHead(nn.Module):
    """Unified regression head for predicting positive scalars (rates, lengths, etc.).

    Supports two output modes:
      - "continuous": activation function + affine scaling  ->  scalar per token
      - "discretized": softmax over bins  ->  expected-value scalar per token

    Optional AdaLN conditioning and MLP vs single-linear architecture.
    """

    def __init__(
        self,
        d_model: int,
        head_type: Literal["continuous", "discretized"] = "continuous",
        scalar_fn: Literal["softplus", "exp", "sigmoid", "identity"] = "softplus",
        softplus_beta: float = 1.0,
        init_bias: Optional[float] = -4.0,
        num_bins: int = 100,
        min_val: float = 0.0,
        max_val: float = 1.0,
        use_mlp: bool = True,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.head_type = head_type
        self.scalar_fn = scalar_fn
        self.min_val = min_val
        self.max_val = max_val
        self.use_mlp = use_mlp
        self.has_cond = cond_dim is not None

        self.norm = LayerNormAndScale(d_model)

        if self.has_cond:
            self.adaLN_modulation = AdaLNModulations(
                cond_dim, d_model, num_modulation_parameters=2
            )

        if use_mlp:
            self.proj1 = nn.Linear(d_model, d_model)
            self.act = nn.GELU()
            hidden_dim = d_model
        else:
            hidden_dim = d_model

        if head_type == "continuous":
            self.proj2 = nn.Linear(hidden_dim, 1)

            if scalar_fn == "softplus":
                self.fn = nn.Softplus(beta=softplus_beta)
                if init_bias is not None:
                    with torch.no_grad():
                        self.proj2.bias.fill_(init_bias)
            elif scalar_fn == "exp":
                self.fn = torch.exp
            elif scalar_fn == "sigmoid":
                self.fn = nn.Sigmoid()
            elif scalar_fn != "identity":
                raise ValueError(f"Unknown scalar_fn: {scalar_fn}")

        elif head_type == "discretized":
            self.proj2 = nn.Linear(hidden_dim, num_bins)
            self.num_bins = num_bins
            self.register_buffer(
                "bin_centers",
                torch.linspace(min_val, max_val, num_bins),
                persistent=True,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()

        x = self.norm(x)
        if self.has_cond and c is not None:
            c = c.float()
            shift, scale = self.adaLN_modulation(c)
            x = AdaLNModulations.ada_ln_modulate(x, shift, scale)

        if self.use_mlp:
            x = self.act(self.proj1(x))

        s = self.proj2(x)

        if self.head_type == "continuous":
            if self.scalar_fn == "identity":
                out = s.squeeze(-1)
            else:
                out = (
                    self.fn(s).squeeze(-1) * (self.max_val - self.min_val)
                    + self.min_val
                )
        else:
            probs = torch.nn.functional.softmax(s, dim=-1)
            out = (probs * self.bin_centers).sum(dim=-1)

        return out.to(orig_dtype)
