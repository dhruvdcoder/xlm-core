from typing import Optional
import torch
from torch import Tensor as TT
import torch.nn as nn
import math
from jaxtyping import Integer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # max_len x 1
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )  # exp(d_vec * -log(10000) / d_model) where d_vec is
        # the vector of frequencies [0, 2, ..., d_model]
        # div_term[i] = exp(-2i log(10000) / d_model),
        #  where i goes from 0 to d_model/2 - 2 (incl.)
        # so div_term starts with exp(0) = 1 and decays exponentially
        # to exp(-log(10000)) = 1/10000
        # in sin(Omega*t) position is t and entries of div_term are different angular frequencies
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # shape (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # shape (max_len, d_model/2)
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[: x.size(0), :]


class _RotaryEmbedding_CompileUnfriendly(nn.Module):
    def __init__(
        self, dim: int, head_first: bool = True, cache_size: int = 1024
    ):
        """
        Args:
            dim: the dimension of the input.
            head_first: if True, the input is assumed to be of shape (batch_size, seq_len, num_heads, dim)
                        if False, the input is assumed to be of shape (batch_size, num_heads, seq_len, dim)
            cache_size: the maximum sequence length to cache the sine and cosine values for.
        """
        super().__init__()
        self.dim = dim
        self.head_first = head_first
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # shape (dim/2)
        angles = self.get_angles(cache_size)
        self.register_buffer(
            "sin", angles.sin(), persistent=False
        )  # shape (seq_len, dim/2), persistent=False to avoid saving to checkpoint, which may cause size mismatch while loading
        self.register_buffer(
            "cos", angles.cos(), persistent=False
        )  # shape (seq_len, dim/2)
        self.seq_len = cache_size

    def forward(
        self, x, positions: Optional[Integer[TT, " *batch seq_len"]] = None
    ):
        """
        Args:
            x: shape (batch_size, seq_len, num_heads, dim) if head_first is False
               shape (batch_size, num_heads, seq_len, dim) if head_first is True
            positions: shape (batch_size, seq_len)
        """
        if positions is None:
            seq_len = x.shape[2] if self.head_first else x.shape[1]
        else:
            seq_len = (
                int(torch.max(positions).item()) + 1
            )  # .item() is compile unfriendly

        # Don't update the buffers. It is compile unfriendly.
        # if seq_len > self.seq_len:
        #    # Update the buffers
        #    t = self.get_angles(seq_len)
        #    self.sin = t.sin()  # shape (seq_len, dim/2)
        #    self.cos = t.cos()  # shape (seq_len, dim/2)
        #    self.seq_len = seq_len

        if not self.head_first:
            if positions is None:
                sin = self.sin[
                    None, :seq_len, None, :
                ]  # shape (1, seq_len, 1, dim/2)
                cos = self.cos[
                    None, :seq_len, None, :
                ]  # shape (1, seq_len, 1, dim/2)
            else:
                sin = torch.nn.functional.embedding(
                    positions, self.sin
                ).unsqueeze(
                    -2
                )  # shape (batch_size, seq_len, 1, dim/2)
                cos = torch.nn.functional.embedding(
                    positions, self.cos
                ).unsqueeze(
                    -2
                )  # shape (batch_size, seq_len, 1, dim/2)
        else:
            if positions is None:
                sin = self.sin[
                    None, None, :seq_len, :
                ]  # shape (1, 1, seq_len, dim/2)
                cos = self.cos[
                    None, None, :seq_len, :
                ]  # shape (1, 1, seq_len, dim/2)
            else:
                sin = torch.nn.functional.embedding(
                    positions, self.sin
                ).unsqueeze(
                    -3
                )  # shape (batch_size, 1, seq_len, dim/2)
                cos = torch.nn.functional.embedding(
                    positions, self.cos
                ).unsqueeze(
                    -3
                )  # shape (batch_size, 1, seq_len, dim/2)

        x_rope_1, x_rope_2 = x.chunk(
            2, dim=-1
        )  # shape (bsz, num_heads, seq_len, dim/2)
        x_rotated = torch.cat(
            [x_rope_1 * cos - x_rope_2 * sin, x_rope_2 * cos + x_rope_1 * sin],
            dim=-1,
        )
        return x_rotated

    def get_angles(self, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(
            self.inv_freq
        )
        angles = torch.einsum(
            "i,j->ij", t, self.inv_freq
        )  # shape (seq_len, dim/2)
        return angles


# compile friendly version
class RotaryEmbedding(nn.Module):
    def __init__(
        self, dim: int, head_first: bool = True, cache_size: int = 1024
    ):
        """
        Args:
            dim: the dimension of the input.
            head_first: if True, the input is assumed to be of shape (batch_size, seq_len, num_heads, dim)
                        if False, the input is assumed to be of shape (batch_size, num_heads, seq_len, dim)
            cache_size: the maximum sequence length to cache the sine and cosine values for.
        """
        super().__init__()
        self.dim = dim
        self.head_first = head_first
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # shape (dim/2)
        # During to an ideosyncrasy of generation code (an empty/mask token can be added on the rightmost),
        # we need to keep one position
        # more than the max length.
        cache_size += 1
        angles = self.get_angles(cache_size)
        self.register_buffer(
            "sin", angles.sin(), persistent=False
        )  # shape (seq_len, dim/2), persistent=False to avoid saving to checkpoint, which may cause size mismatch while loading
        self.register_buffer(
            "cos", angles.cos(), persistent=False
        )  # shape (seq_len, dim/2)
        self.seq_len = cache_size

    def forward(self, x, positions: Integer[TT, " *batch seq_len"]):
        """
        Args:
            x: shape (batch_size, seq_len, num_heads, dim) if head_first is False
               shape (batch_size, num_heads, seq_len, dim) if head_first is True
            positions: shape (batch_size, seq_len)
        """

        if not self.head_first:
            sin = torch.nn.functional.embedding(positions, self.sin).unsqueeze(
                -2
            )  # shape (batch_size, seq_len, 1, dim/2)
            cos = torch.nn.functional.embedding(positions, self.cos).unsqueeze(
                -2
            )  # shape (batch_size, seq_len, 1, dim/2)
        else:
            sin = torch.nn.functional.embedding(positions, self.sin).unsqueeze(
                -3
            )  # shape (batch_size, 1, seq_len, dim/2)
            cos = torch.nn.functional.embedding(positions, self.cos).unsqueeze(
                -3
            )  # shape (batch_size, 1, seq_len, dim/2)

        x_rope_1, x_rope_2 = x.chunk(
            2, dim=-1
        )  # shape (bsz, num_heads, seq_len, dim/2)
        x_rotated = torch.cat(
            [x_rope_1 * cos - x_rope_2 * sin, x_rope_2 * cos + x_rope_1 * sin],
            dim=-1,
        )
        return x_rotated

    def get_angles(self, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(
            self.inv_freq
        )
        angles = torch.einsum(
            "i,j->ij", t, self.inv_freq
        )  # shape (seq_len, dim/2)
        return angles
