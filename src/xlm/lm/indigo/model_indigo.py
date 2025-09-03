from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT

from xlm.modules.rotary_transformer import (
    RotaryTransformerFinalLayer,
    RotaryTransformerFinalLayerForClassification,
    RotaryTransformerLayer,
    RotaryTransformerLayerList,
    RotaryEmbedding,
)
from xlm.modules.gpt2_transformer import GPT, GPTConfig

from torch.nn.attention import SDPBackend

from xlm.model import Model
from xlm.utils.rank_zero import RankedLogger
from .types_indigo import IndigoBatch
import math


logger = RankedLogger(__name__, rank_zero_only=True)


# TODO (URV): Implement a new transformer model that uses tertiary positional embeddings like shown in the indigo paper.
# you can use the rotary transformer for reference but the way you encode the tertiary positional embeddings will be different.

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
# region: Indigo Transformer


class IndigoModel(nn.Module):
    """Indigo model implementation."""

    def __init__( 
            self,
        ):

        super().__init__()
        # TODO (URV): Initialize your model components
       
       # Initialize other things among the two layers below
       # Get list of IndigoTransformerLayer from IndigoTransformerLayerList here
       # Get the IndigoTransformerFinalLayer
       # Get the IndigoTransformerPointerNetwork layer initialized 
       # get_position would use the IndigoTransformerPointerNetwork, it wont be used in forward
       # IndigoTransformerLayer, IndigoTransformerLayerList, IndigoTransformerFinalLayer are used in forward
       
        

    def forward(
        self,
        input_ids: Integer[TT, " batch seq_len"],
        attention_mask: Optional[Integer[TT, " batch seq_len"]] = None,
        **kwargs,
    ):
        # TODO (URV): Implement your forward pass
        pass
    
    def get_position(self):
        pass

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


# endregion: Indigo Transformer
########################################################

class IndigoTransformerLayer(nn.Module):
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
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = dropout
        self.head_dim = d_model // nhead


        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False) # gets you the Q,K,V matrices
        #self.relative_position_emb_A = nn.Parameter(torch.randn(3, d_model // nhead)) #Matrix A from the paper (3, d_head)
        self.relative_position_emb_A = torch.nn.Embedding(3, d_model // nhead)  # Matrix A from the paper (3, d_head)

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
            self,
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
            q, # shape:(bsz, nheads, seqlen, head_dim)
            k, # shape:(bsz, nheads, seqlen, head_dim)
            v, # shape:(bsz, nheads, seqlen, head_dim)
            rel_matrix, # shape:(bsz, seqlen, seqlen)
            attn_mask, # shape:?
    ):
        # Calculating attention logits
        # We cannot use normal attention method from torch because here for each query vector, we have a different key vector to multiply, depending on their relation. We multiply a single query with a matrix of K for efficiency. For one Q_i in Indigo Attention, you need to mutliply it with a different K matrix, this K matrix is specific to that ith query vector. We first transform the relative matrix, hold the specific vectors from embedding rather than the relative positions. Then we when we add rel_emb and k, we have two broadcast rules. One deals with applying the same tertiary embedding structure to all heads, and the other deals with adding the K matrix to differnt tertiary positions of the Q_i vector possibilies.     
       
        # replaces -1, 0 or 1 with one of the 3 vectors in embedding. Encodes tertiary embeddings
        # rel_emb shape: (bsz, seqlen, seqlen, head_dim)
        rel_emb = self.relative_position_emb_A(rel_matrix + 1)

        # rel_emb shape: (bsz, 1, seqlen, seqlen, head_dim)
        rel_emb = rel_emb.unsqueeze(1)

        #k shape: (bsz, nheads, 1, seqlen, head_dim)
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
        # Here after squeezing the useless dimension, we should have to correct logits as in normal attention methods.
        # Now we divide by sqrt of dmodel, apply mask, softmax, matmul with v and apply attention dropout layer and then output logits

        attn_logits = attn_logits / math.sqrt(self.head_dim)
        attn_logits = attn_logits.masked_fill(attn_mask, -math.inf)  # if mask is given
        attn_weights = F.softmax(attn_logits, dim=-1)  # attention scores
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self.dropout_layer(attn_output) # Don't know if this would turn off dropout during inference. 

        # attn_output shape: (bsz, n_heads, seq_len, head_dim)
        return attn_output
        # This returns the logits, and culminates the attention layer, now there would come a pre mlp norm, then mlp/feedforward, then another dropout
        # Then Layer class ends

class IndigoTransformerLayerList(nn.ModuleList):
    def __init__(self):
        pass

    @classmethod
    def from_layer(self):
        pass

class IndigoTransformerFinalLayer(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

class IndigoTransformerPointerNetwork(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass
