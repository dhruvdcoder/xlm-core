import math
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from xlm.utils.nn import get_mask_from_sequence_lengths
from .position import PositionalEncoding

# We will stick to the pytorch api for TransformerEncoderLayer and TransformerEncoder


class DiffusionTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        raise NotImplementedError("Not complete. Don't use this.")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scaling = self.head_dim**-0.5

        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = getattr(F, activation)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.norm_first = norm_first
        self.batch_first = batch_first

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: shape (B, T, d_model) for now we will assume T=query_seq_len=key_seq_len
            src_mask: shape (B*num_heads, T, T) or (T, T). True is attend, False is not attend
                Note that nn.TransformerEncoderLayer will allow float masks which are added to
                the attention scores. But we only support boolean masks here.
            src_key_padding_mask: shape (B, T) or (T). True is attend, False is masked
        """
        # Let's not implement src_mask for now
        if src_mask is not None:
            raise NotImplementedError("src_mask is not implemented")
        x = src  # shape (B, T, d_model)
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask)
            )
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        # x shape (B, T, d_model)
        bsz, seq_len, _ = (
            x.shape
            if self.batch_first
            else (x.shape[1], x.shape[0], x.shape[2])
        )
        q, k, v = self.self_attn_qkv(x).chunk(
            3, dim=-1
        )  # q shape (B, T, d_model)
        q, k, v = map(
            lambda t: t.view(
                bsz, seq_len, self.nhead, self.head_dim
            ).transpose(1, 2),
            (q, k, v),
        )  # q shape (B, nhead, T, head_dim)

        # Combine attn_mask and key_padding_mask
        if key_padding_mask is not None:
            if attn_mask is None and key_padding_mask.dim() == 2:
                attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            else:
                attn_mask = attn_mask.logical_or(
                    key_padding_mask.unsqueeze(1).unsqueeze(2)
                )

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,  # shape (B, query_seq_len, key_seq_len)
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.d_model)
        )
        return self.dropout(self.self_attn_out(attn_output))

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation_fn(self.linear1(x))))
        return self.dropout(x)


class DiffusionTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_source_positions: int = 1024,
    ):
        super().__init__()
        if not batch_first:
            raise NotImplementedError("batch_first=False is not implemented")
        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(
            num_embeddings, d_model, padding_idx=self.padding_idx
        )
        self.embed_positions = PositionalEncoding(
            d_model, max_source_positions
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(
        self, src_tokens: Tensor, src_lengths: Optional[Tensor]
    ) -> dict[str, Tensor]:
        """
        Args:
            src_tokens: shape (B, T)
            src_lengths: shape (B) of type LongTensor
        """
        # Embed tokens and positions
        x = self.embed_tokens(src_tokens) * math.sqrt(
            self.embed_tokens.embedding_dim
        )  # this scaling is done in the original "Attention is All you need"
        # but not in T5 because T5 uses a different position encoding
        x = x + self.embed_positions(src_tokens)
        x = self.dropout(x)

        # Compute padding mask
        if src_lengths is not None:
            encoder_padding_mask = get_mask_from_sequence_lengths(
                src_lengths, max_length=src_tokens.size(1)
            )
        else:
            encoder_padding_mask = src_tokens.eq(
                self.padding_idx
            )  # shape (B, T)

        # Apply transformer encoder
        # if not self.batch_first:
        #    x = x.transpose(0, 1)  # B x T x C -> T x B x C

        x = self.encoder(
            x, src_key_padding_mask=encoder_padding_mask, is_causal=False
        )

        x = self.layer_norm(x)
        x = self.dropout(x)

        # if not self.batch_first:
        #    x = x.transpose(0, 1)  # T x B x C -> B x T x C
        return {"encoder_out": x}
