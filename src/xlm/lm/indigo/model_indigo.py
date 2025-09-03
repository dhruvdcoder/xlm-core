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
        inp: torch.Tensor,
        attention_mask: torch.Tensor,
        rel_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).
            attention_mask: the attention mask of shape (bsz, seq_len), which is True for non-padding tokens.
             It can also be of shape (bsz, seq_len (query), seq_len (key-value)), where the mask indicates which tokens are valid in the context.

             rel_matrix shape would be: (bsz, n_heads, seq_len, seq_len)
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
            rel_matrix,
            attn_mask=attn_mask,
        )
        # shape (bsz, n_heads, seq_len, head_dim)

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
        q,  # shape:(bsz, nheads, seqlen, head_dim)
        k,  # shape:(bsz, nheads, seqlen, head_dim)
        v,  # shape:(bsz, nheads, seqlen, head_dim)
        rel_matrix,  # shape:(bsz, seqlen, seqlen)
        attn_mask,  # shape:?
    ):
        # Calculating attention logits
        # We cannot use normal attention method from torch because here for each query vector, we have a different key vector to multiply, depending on their relation. We multiply a single query with a matrix of K for efficiency. For one Q_i in Indigo Attention, you need to mutliply it with a different K matrix, this K matrix is specific to that ith query vector. We first transform the relative matrix, hold the specific vectors from embedding rather than the relative positions. Then we when we add rel_emb and k, we have two broadcast rules. One deals with applying the same tertiary embedding structure to all heads, and the other deals with adding the K matrix to differnt tertiary positions of the Q_i vector possibilies.

        # replaces -1, 0 or 1 with one of the 3 vectors in embedding. Encodes tertiary embeddings
        # rel_emb shape: (bsz, seqlen, seqlen, head_dim)
        rel_emb = self.relative_position_emb_A(rel_matrix + 1)

        # rel_emb shape: (bsz, 1, seqlen, seqlen, head_dim)
        rel_emb = rel_emb.unsqueeze(1)

        # k shape: (bsz, nheads, 1, seqlen, head_dim)
        k = k.unsqueeze(2)

        # k_aug shape: (bsz, nheads, seqlen, seqlen, head_dim) BROADCASTED
        k_aug = k + rel_emb
        # Here k_aug, for all bsz, for all heads, a 3d tensor. Think of it as a 2d matrix of seqlen * seqlen, and the entries are a vector of head_dim
        # This vector is one of the three from the embedding layer, we basically convert the entire rel_mat from -1,0,1 to the respective vectors,
        # and then we just add it to k via broadcasting.
        # Broadcasts:
        # 1. The same multiple K matrix structure is added to all heads.
        # 2. The same base K matrix is added to differnt versions of Q_i based tertiary embeddings

        # q shape:(bsz, nheads, seqlen, 1, head_dim)
        q = q.unsqueeze(3)

        # attn_logits shape: (bsz, nheads, seqlen, 1, seqlen)
        attn_logits = torch.matmul(q, k_aug.transpose(-2, -1))
        # Every Q_i is matrix multiplied with its corresponding K matrix to get one row of attention logit. Example: Q_3 vector would be matrix multiplied with k_aug[:,:,3] matrix as k_aug[:,:,3] is k[:,:] + A[r_3,j + 1]

        # attn_logits shape was: (bsz, nheads, seqlen, 1, seqlen)
        # attn_logits shape after this step: (bsz, nheads, seqlen, seqlen)
        attn_logits = attn_logits.squeeze(3)

        attn_logits = attn_logits / math.sqrt(self.head_dim)
        attn_logits = attn_logits.masked_fill(attn_mask, -math.inf)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self.dropout_layer(
            attn_output
        )  # Don't know if this would turn off dropout during inference.

        # attn_output shape: (bsz, n_heads, seq_len, head_dim)
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
        self.pointer_projection = nn.Linear(
            3 * d_model, d_model, bias=False
        )  # joint matrix for C,D,E projections

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        pi: Optional[Integer[TT, " *batch seq_len"]] = None,
        attention_mask: Optional[Bool[TT, " *batch seq_len seq_len"]] = None,
        rel_matrix: Optional[Integer[TT, " *batch seq_len seq_len"]] = None,
    ):
        if rel_matrix is None and pi is None:
            raise ValueError("rel_matrix or pi must be provided")
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        x = self.embed_tokens(x_t)  # shape (batch_size, seq_len, d_model)
        if rel_matrix is None:
            rel_matrix = get_tertiary_relative_position_matrix(pi)

        for block in self.encoder:
            x = block(x, rel_matrix, attention_mask)

        vocab_logits = self.output_layer(
            x,
        )  # shape (batch_size, seq_len, vocab_size)
        return x, vocab_logits

    def get_position_logits(
        self,
        hidden_states: Float[TT, " *batch seq_len d_model"],
        target_ids: Integer[TT, " *batch seq_len"],
    ) -> Float[TT, " *batch seq_len 2 seq_len"]:
        """
        Args:
            hidden_states: Hidden states produced by the transformer.
            target_ids: Target token ids for which  we are trying to find the insertion position.

        Returns:
            logits: Logits for the position prediction. Note that these logits need to be masked before applying softmax because some positions are in the prompt (constrained) or in the future (during training).
        """
        # seq_len = target_ids.size(1)
        # d_model = hidden_states.size(-1)
        d_model = self.d_model
        embed_matrix = self.embed_tokens(
            target_ids
        )  # shape (*batch_size, seq_len, d_model)
        proj = self.pointer_projection(
            hidden_states
        )  # shape (*batch_size, seq_len, 3*d_model)

        queries = (
            proj[..., :d_model] + embed_matrix
        )  # shape (*batch_size, query_seq_len, d_model)
        keys = proj[..., d_model:].view(
            *proj.shape[:-1], 2, d_model
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
