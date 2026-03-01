"""Modules for simple transformer decoder that uses rotary embeddings for positional encoding."""

import copy
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
import math
from omegaconf import DictConfig

from xlm.modules.position import RotaryEmbedding
from xlm.utils.nn import get_autocast_dtype
import logging

logger = logging.getLogger(__name__)


def add_bias_apply_dropout_scale(
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scale: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    training: bool = True,
) -> torch.Tensor:
    """
    Adds bias, applies dropout, scales, and adds residual.

    TODO: Consider creating fused implementation using jit and two wrappers
    Args:
        x: The input tensor of shape (bsz, seq_len, dim).
        bias: The bias tensor of shape (bsz, 1, dim).
        dropout: The dropout rate.
        scale: The scale tensor of shape (bsz, 1, dim).
        residual: The residual tensor of shape (bsz, seq_len, dim).

    Returns:
        The output tensor of shape (bsz, seq_len, dim).
    """
    x = x + bias if bias is not None else x
    x = F.dropout(x, p=dropout, training=training) if dropout > 0.0 else x
    x = x * scale if scale is not None else x
    x = x + residual if residual is not None else x
    return x


#################################################################################
#                                 Core Model                                    #
#################################################################################


class RotaryTransformerLayer(nn.Module):
    """One layer of DDiT.

    It consists of a multi-head self-attention layer followed by a feedforward layer with normalization and gating in between.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        force_flash_attn: bool = False,
    ):
        """
        Initialize the DDiTBlock.

        Args:
            d_model: the dimension of the input.
            nhead: the number of attention heads.
            mlp_ratio: the ratio of the hidden size of the MLP/feedforward layer to the input size.
            dropout: the dropout rate.
        """
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.n_heads = nhead
        self.dim = d_model
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = dropout
        self.head_dim = d_model // nhead

        # self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim)
        self.rotary_emb = None

        # Single QKV projection
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # TODO: consider using FusedMLP from flash_attn here
        if activation == "gelu":
            act = nn.GELU(approximate="tanh")
        elif activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError(f"Activation {activation} not supported")
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=True),
            act,
            nn.Linear(dim_feedforward, d_model, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout
        self.force_flash_attn = force_flash_attn
        if force_flash_attn:
            self.attn_backend = [SDPBackend.FLASH_ATTENTION]
        else:
            # let torch choose
            self.attn_backend = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
            ]

    def set_rotary_emb(self, rotary_emb: Optional[RotaryEmbedding] = None):
        if rotary_emb is None:
            logger.info(
                "RotaryEmbedding not provided. Using default with size=head_dim"
            )
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        else:
            self.rotary_emb = rotary_emb
        self.rotary_emb_dim = rotary_emb.dim
        if self.rotary_emb_dim > self.head_dim:
            raise ValueError(
                "RotaryEmbedding dimension is greater than the head dimension."
            )

    def forward(
        self,
        inp: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).
            attention_mask: the attention mask of shape (bsz, seq_len), which is True for non-padding tokens.
             It can also be of shape (bsz, seq_len (query), seq_len (key-value)), where the mask indicates which tokens are valid in the context.
        """
        if self.rotary_emb is None:
            raise ValueError(
                "RotaryEmbedding is not set. Call set_rotary_emb() to set it."
            )

        # region: attention ie: inp = dropout(attn(norm(inp))) + inp
        # region: prenorm. Apply normalization before the attention
        x = self.norm1(inp)
        # endregion: prenorm

        # region: attention+position
        # Generate rotary position embeddings
        seq_len = x.shape[1]
        # rotary_pos_emb = self.rotary_emb(seq_len, x.device)

        # Project to q, k, v
        qkv = self.attn_qkv(x)  # shape (bsz, seq_len, 3 * dim)
        q, k, v = qkv.chunk(3, dim=-1)  # shape (bsz, seq_len, dim)

        # Reshape to (batch_size, n_heads, seq_len, head_dim)
        q = q.view(
            q.shape[0], q.shape[1], self.n_heads, self.head_dim
        ).transpose(
            1, 2
        )  # shape (bsz, n_heads, seq_len, head_dim)
        k = k.view(
            k.shape[0], k.shape[1], self.n_heads, self.head_dim
        ).transpose(
            1, 2
        )  # shape (bsz, n_heads, seq_len, head_dim)
        v = v.view(
            v.shape[0], v.shape[1], self.n_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary embeddings to q and k
        q_rotary = self.apply_rotary_pos_emb(
            q, positions
        )  # shape (bsz, n_heads, seq_len, head_dim)
        k_rotary = self.apply_rotary_pos_emb(
            k, positions
        )  # shape (bsz, n_heads, seq_len, head_dim)

        # Perform scaled dot-product attention
        # Make the attention mask broadcastable to (bsz, query_seq_len(1), key_seq_len(seq_len))
        # Note we want to broadcast (copy) along the query_seq_len dimension
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:  # (bsz, seq_len)
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(
                    1
                )  # shape (bsz, 1, 1, seq_len)
            elif (
                attention_mask.ndim == 3
            ):  # (bsz, seq_len (query), seq_len (key-value))
                attn_mask = attention_mask.unsqueeze(
                    1
                )  # shape (bsz, 1, seq_len (query), seq_len (key-value))
            else:
                raise ValueError(
                    f"Attention mask must be of shape (bsz, seq_len) or (bsz, seq_len (query), seq_len (key-value)). Got {attention_mask.shape}"
                )

        # UPGRADE: The following context manager is not compile friendly
        # upto torch 2.5.1. It can be compiled with torch 2.6.0, but
        # due to the new default of `torch.load(weights_only=True)`,
        # torch 2.6.0 will not work with lightning 2.3, 2.4 or 2.5.
        # So untill lightning supports torch 2.6, we cannot use this context manager.
        with torch.nn.attention.sdpa_kernel(self.attn_backend):
           attn_output = F.scaled_dot_product_attention(
               q_rotary,
               k_rotary,
               v,
               attn_mask=attn_mask,
               dropout_p=self.dropout if self.training else 0.0,
           )  # shape (bsz, n_heads, seq_len, head_dim)

        #attn_output = F.scaled_dot_product_attention(
        #    q_rotary,
        #    k_rotary,
        #    v,
        #    attn_mask=attn_mask,
        #    dropout_p=self.dropout if self.training else 0.0,
        #)  # shape (bsz, n_heads, seq_len, head_dim)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], seq_len, self.dim)
        )  # shape (bsz, seq_len, dim)
        x_attn = self.o_proj(attn_output)  # shape (bsz, seq_len, dim)
        # endregion: attention + position

        # region: attention residual connection
        inp = add_bias_apply_dropout_scale(
            x_attn,
            bias=None,
            dropout=self.dropout,
            scale=None,
            residual=inp,
            training=self.training,
        )
        # endregion: attention residual connection
        # endregion: attention

        # region: feedforward ie: inp = dropout(mlp(norm(inp))) + inp
        # norm -> MLP -> dropout -> residual
        inp = add_bias_apply_dropout_scale(
            self.mlp(self.norm2(inp)),
            bias=None,
            dropout=self.dropout,
            scale=None,
            residual=inp,
            training=self.training,
        )
        # endregion: feedforward
        return inp

    def apply_rotary_pos_emb(
        self, x, positions: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: the input tensor of shape (batch_size, seq_len, num_heads, dim).

        Returns:
            The tensor with rotary position embeddings applied to the first dim/2 of the last dimension.
        """
        x_rope = x[
            ..., : self.rotary_emb_dim
        ]  # shape (bsz, seq_len, n_heads, dim/2)
        x_pass = x[..., self.rotary_emb_dim :]
        x_rotated = self.rotary_emb(x_rope, positions)  # type: ignore
        return torch.cat([x_rotated, x_pass], dim=-1)


class RotaryTransformerLayerList(nn.ModuleList):
    """A module list of DDiT blocks that share the rotary cache for the rotary embeddings."""

    def __init__(
        self, blocks: List[RotaryTransformerLayer], rotary_emb: RotaryEmbedding
    ):

        for block in blocks:
            block.set_rotary_emb(rotary_emb)
        super().__init__(blocks)

    @classmethod
    def from_layer(
        cls,
        layer: RotaryTransformerLayer,
        num_layers: int,
        rotary_emb: RotaryEmbedding,
    ):
        return cls(
            [copy.deepcopy(layer) for _ in range(num_layers)], rotary_emb
        )


class RotaryTransformerFinalLayer(nn.Module):
    """Simple unembedding layer with optional layer norm."""

    def __init__(
        self,
        d_model: int,
        out_dims: int,
        layer_norm_eps: float = 1e-5,
        use_final_layer_norm: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        self.norm_final = (
            nn.LayerNorm(d_model, eps=layer_norm_eps)
            if use_final_layer_norm
            else None
        )
        self.linear = nn.Linear(d_model, out_dims, bias=False)
        if zero_init:
            with torch.no_grad():
                self.linear.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).
        """
        if self.norm_final is not None:
            x = self.norm_final(x)
        x = self.linear(x)
        return x


class RotaryTransformerFinalLayerForClassification(nn.Module):
    """Feedforward layer with pre-norm and residual connection followed by a linear layer for classification."""

    def __init__(
        self,
        d_model: int,
        out_dims: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=True),
            nn.Tanh(),
            nn.Linear(2 * d_model, d_model, bias=True),
        )
        self.linear = nn.Linear(d_model, out_dims, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).
        """
        x = add_bias_apply_dropout_scale(
            self.mlp(self.norm_final(x)),
            bias=None,
            dropout=self.dropout,
            scale=None,
            residual=x,
            training=self.training,
        )

        x = self.linear(x)
        return x
