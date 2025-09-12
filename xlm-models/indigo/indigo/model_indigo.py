from typing import Optional, Tuple, Dict, Any
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT
from typing import List, Optional
from torch.nn.attention import SDPBackend
from xlm.utils.rank_zero import RankedLogger
from .types_indigo import IndigoBatch
import math
from .utils import get_tertiary_relative_position_matrix


logger = RankedLogger(__name__, rank_zero_only=True)


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


########################################################
# region: Indigo Transformer Utility Classes


class IndigoTransformerLayer(nn.Module):
    """Implements tertiary position based attention."""

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
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.n_heads = nhead
        self.dim = d_model
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = dropout
        self.head_dim = d_model // nhead

        self.attn_qkv = nn.Linear(
            d_model, 3 * d_model, bias=False
        )  # gets you the Q,K,V matrices
        # TODO: Should we have separate relative position embeddings for each head?
        self.relative_position_emb_A = torch.nn.Embedding(
            3, d_model // nhead
        )  # Matrix A from the paper (3, d_head)

        self.o_proj = nn.Linear(d_model, d_model, bias=False)

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
        self.dropout_layer = nn.Dropout(dropout)

        if force_flash_attn:
            self.attn_backend = [SDPBackend.FLASH_ATTENTION]
        else:
            # let torch choose
            self.attn_backend = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
            ]

    def forward(
        self,
        inp: Float[TT, " bsz query_len dim"],
        attention_mask: torch.Tensor,
        rel_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, query_len, dim).
            attention_mask: the attention mask of shape (bsz, seq_len), which is True for non-padding tokens.
             It can also be of shape (bsz, seq_len (query), seq_len (key-value)), where the mask indicates which tokens are valid in the context.
            rel_matrix shape would be: (bsz, n_heads, key_seq_len, query_seq_len)
        """

        # region: attention ie: inp = dropout(attn(norm(inp))) + inp
        # region: prenorm. Apply normalization before the attention
        x = self.norm1(inp)
        # endregion: prenorm

        # region: attention+position
        # Generate rotary position embeddings
        seq_len = x.shape[1]

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

        # Perform scaled dot-product attention
        # Make the attention mask broadcastable to (bsz, query_seq_len(1), key_seq_len(seq_len))
        # Note we want to broadcast (copy) along the query_seq_len dimension
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

        attn_output = self.indigo_scaled_dot_product_attention(
            q,
            k,
            v,
            rel_matrix.transpose(-1, -2),
            attn_mask=attn_mask,
        )
        # shape (bsz, n_heads, query_len, head_dim)

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

    def indigo_scaled_dot_product_attention(
        self,
        q: Float[TT, " bsz nheads query_len head_dim"],
        k: Float[TT, " bsz nheads key_len head_dim"],
        v: Float[TT, " bsz nheads key_len head_dim"],
        rel_matrix: Float[TT, " bsz query_len key_len"],
        attn_mask: Bool[TT, " bsz nheads query_len key_len"],
    ) -> Float[TT, " bsz nheads query_len head_dim"]:
        # ternary relative embedding: (-1,0,1) -> vectors in R^{head_dim}
        # rel_emb: (bsz, query_len, key_len, head_dim)
        rel_emb = self.relative_position_emb_A(rel_matrix + 1)

        # share same rel embeddings across heads (add a head dim of 1)
        rel_emb = rel_emb.unsqueeze(
            1
        )  # (bsz, 1, query_len, key_len, head_dim)

        # expand keys over query positions, then add relative embedding
        k_aug = (
            k.unsqueeze(-3) + rel_emb
        )  # (bsz, nheads, query_len, key_len, head_dim)

        # dot(q, k_aug) over head_dim -> logits with shape (bsz, nheads, query_len, key_len)
        # NOTE: no extra unsqueeze on q needed
        attn_logits = torch.einsum(
            "bhqkd,bhqd->bhqk", k_aug, q
        )  # (bsz, nheads, query_len, key_len)

        # scale
        attn_logits = attn_logits / math.sqrt(
            self.head_dim
        )  # TODO: Use tensor instead of raw float in the division.

        # ensure bool, broadcastable to (bsz, nheads, query_len, key_len)
        attn_logits = attn_logits.masked_fill(~attn_mask, -torch.inf)

        # attention weights over keys
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # weighted sum of values
        attn_output = torch.matmul(
            attn_weights, v
        )  # (bsz, nheads, query_len, head_dim)
        return attn_output


class IndigoTransformerLayerList(nn.ModuleList):
    def __init__(self, blocks: List[IndigoTransformerLayer]):
        super().__init__(blocks)

    @classmethod
    def from_layer(
        cls,
        layer: IndigoTransformerLayer,
        num_layers: int,
    ):
        return cls([copy.deepcopy(layer) for _ in range(num_layers)])


class IndigoTransformerFinalLayer(nn.Module):
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


# endregion: Indigo Transformer Utility Classes
########################################################

########################################################
# region: Indigo Transformer Model


class IndigoModel(nn.Module):
    """Indigo model implementation."""

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
        max_length: int = 1024,
        force_flash_attn: bool = False,
        final_layer_without_normalization: bool = False,
    ):

        # TODO (URV): Initialize your model components

        # Initialize other things among the two layers below
        # Get list of IndigoTransformerLayer from IndigoTransformerLayerList here
        # Get the IndigoTransformerFinalLayer
        # Get the IndigoTransformerPointerNetwork layer initialized
        # get_position would use the IndigoTransformerPointerNetwork, it wont be used in forward
        # IndigoTransformerLayer, IndigoTransformerLayerList, IndigoTransformerFinalLayer are used in forward

        super().__init__()
        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(
            num_embeddings, d_model, padding_idx=padding_idx
        )
        self.dim_feedforward = dim_feedforward or 4 * d_model
        self.d_model = d_model
        encoder_layer = IndigoTransformerLayer(
            d_model,
            nhead,
            self.dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            force_flash_attn=force_flash_attn,
        )

        self.max_length = max_length
        self.encoder = IndigoTransformerLayerList.from_layer(
            encoder_layer,
            num_layers,
        )
        self.output_layer = IndigoTransformerFinalLayer(
            d_model,
            num_embeddings,
            layer_norm_eps,
            use_final_layer_norm=not final_layer_without_normalization,
            zero_init=False,
        )
        self.pointer_projection_query = nn.Linear(
            d_model, d_model, bias=False
        )  # joint matrix for E projections
        self.pointer_projection_key = nn.Linear(
            d_model, 2 * d_model, bias=False
        )  # joint matrix for C,D projections

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        pi: Optional[Integer[TT, " *batch seq_len"]] = None,
        attention_mask: Optional[Bool[TT, " *batch query_len key_len"]] = None,
        rel_matrix: Optional[Integer[TT, " *batch key_len query_len"]] = None,
    ):
        if rel_matrix is None and pi is None:
            raise ValueError("rel_matrix or pi must be provided")
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        # check the shape of attention_mask and expand to (bsz, query_len(1), key_len) if needed
        # also make it lower triangular
        if attention_mask.ndim == 2:
            key_len = attention_mask.size(-1)
            query_len = key_len
            causal_attention_mask = torch.tril(
                torch.ones(
                    query_len,
                    key_len,
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
            )
            attention_mask = (
                attention_mask.unsqueeze(1) & causal_attention_mask
            )

        x = self.embed_tokens(x_t)  # shape (batch_size, seq_len, d_model)
        if rel_matrix is None:
            assert pi is not None
            rel_matrix = get_tertiary_relative_position_matrix(pi)

        for block in self.encoder:
            x = block(x, attention_mask, rel_matrix)

        vocab_logits = self.output_layer(
            x,
        )  # shape (batch_size, seq_len, vocab_size)
        return x, vocab_logits

    def get_position_logits(
        self,
        hidden_states: Float[TT, " *batch key_seq_len d_model"],
        target_ids: Integer[TT, " *batch query_seq_len"],
        target_hidden_states: Optional[
            Float[TT, " *batch query_seq_len d_model"]
        ] = None,
    ) -> Float[TT, " *batch seq_len 2 seq_len"]:
        """
        Args:
            hidden_states: Hidden states produced by the transformer.
            target_ids: Target token ids for which  we are trying to find the insertion position.

        Returns:
            logits: Logits for the position prediction. Note that these logits need to be masked before applying softmax because some positions are in the prompt (constrained) or in the future (during training).
        """
        if target_hidden_states is None:
            target_hidden_states = hidden_states
        d_model = self.d_model
        embed_matrix = self.embed_tokens(
            target_ids
        )  # shape (*batch_size, query_seq_len, d_model)
        proj_key = self.pointer_projection_key(
            hidden_states
        )  # shape (*batch_size, key_seq_len, 2*d_model)
        proj_query = self.pointer_projection_query(
            target_hidden_states
        )  # shape (*batch_size, query_seq_len, d_model)

        queries = (
            proj_query + embed_matrix
        )  # shape (*batch_size, query_seq_len, d_model)
        keys = proj_key.view(
            *proj_key.shape[:-1], 2, d_model
        )  # shape(*batch_size, key_seq_len, 2, d_model)
        # b,k,q,x,d = batch, key_seq_len, query_seq_len, 2, d_model
        # Note: for us query_seq_len = key_seq_len = seq_len
        logits = torch.einsum(
            "...kxd,...qd->...kxq", keys, queries
        )  # shape(batch_size, key_seq_len, 2, query_seq_len)
        return logits

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


# endregion: Indigo Transformer Model
########################################################
