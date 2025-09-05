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

# DTYPE = get_autocast_dtype()


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNormAndScale(nn.Module):
    """Performs normalization and just scaling (no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: the dimension of the input.
        """
        super().__init__()
        self.norm = nn.Parameter(
            torch.ones([dim])
        )  # name is norm so that weight decay doesn't apply
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).

        Returns:
            the normalized and scaled output tensor of shape (bsz, seq_len, dim).
        """
        x = F.layer_norm(x, [self.dim], eps=self.eps)
        return x * self.norm[None, None, :]


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
    ):
        """
        Args:
            hidden_size: The size of the hidden layer and the output of MLP.
            frequency_embedding_size: The size of the frequency embedding layer.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        )  # shape (frequency_embedding_size // 2,)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embeds scalar timesteps into vector representations.

        Args:
            t: A 1-D Tensor of bsz indices, one per batch element. These may be fractional.

        Returns:
            An (bsz, hidden_size) Tensor of positional embeddings.
        """
        args = (
            t[:, None].to(dtype=self.freqs.dtype) * self.freqs[None]
        )  # shape (bsz, dim // 2)
        embedding = torch.cat(
            [torch.cos(args), torch.sin(args)], dim=-1
        )  # shape (bsz, frequency_embedding_size)
        t_rep = self.mlp(embedding)  # shape (bsz, hidden_size)
        return t_rep


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.

    Args:
        num_classes (int): The number of classes.
        cond_size (int): The size of the conditioning input.
        label_dropout (Optional[float]): The dropout rate for class labels during training.

    Attributes:
        embedding_table (nn.Embedding): The embedding table for class labels.
        num_classes (int): The number of classes.

    """

    def __init__(
        self,
        num_classes: int,
        cond_size: int,
        label_dropout: Optional[float] = None,
    ):
        super().__init__()
        # have a special embedding at the end to represent absence of a label,
        # which will be used when a training label is dropped out
        assert label_dropout is None or 0 <= label_dropout < 1
        n = num_classes + 1 if label_dropout is not None else num_classes
        self.embedding = nn.Embedding(n, cond_size)
        self.num_classes = num_classes
        self.label_dropout = label_dropout
        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Drop out class labels during training.

        Args:
            labels (torch.Tensor): The input tensor of class labels of shape (bsz,).

        Returns:
            torch.Tensor: The modified class labels with some labels dropped by setting to the missing (last label).
        """
        if self.label_dropout is not None and self.training:
            mask = torch.rand_like(labels.float()) < self.label_dropout
            # set the dropped labels to the last class that represents absence of a label
            labels = torch.where(mask, self.num_classes, labels)
        return labels

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LabelEmbedder module.

        Args:
            labels (torch.Tensor): The input tensor of class labels of shape (bsz,).

        Returns:
            torch.Tensor: The embedded vector representations of the class labels.

        """
        labels = self.drop_labels(labels)
        embeddings = self.embedding(labels)
        return embeddings


class AdaLNModulations(nn.Module):
    """
    Produces the modulation parameters for AdaLN.
    """

    def __init__(
        self, cond_dim: int, dim: int, num_modulation_parameters: int = 6
    ):
        """
        Initializes the AdaLNModulations module.

        Args:
            cond_dim (int): The dimension of the conditioning input.
            dim (int): The hidden size.
        """
        super().__init__()
        self.num_modulation_parameters = num_modulation_parameters
        self.modulation = nn.Linear(
            cond_dim, num_modulation_parameters * dim, bias=True
        )
        self.modulation.weight.data.zero_()
        self.modulation.bias.data.zero_()

    def forward(self, c: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass of the AdaLNModulations module.

        Args:
            c (torch.Tensor): The conditioning input tensor.

        Returns:
            Tuple[torch.Tensor]: The modulation parameters for AdaLN.
                Each tensor has shape (bsz, 1, dim). When num_modulation_paramters=6, these tensors stand for
                the shift and scale parameters for the MHA and MLP layers, and the gating parameters:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
        """
        # Apply the linear layer to get output of shape (bsz, 6 * dim).
        # Then add one dimension to the output to get shape (bsz, 1, 6 * dim).
        # Finally, chunk the output into 6 tensors of shape (bsz, 1, dim).
        return self.modulation(c)[:, None].chunk(
            self.num_modulation_parameters, dim=2
        )

    # add jit.script to make it faster ?
    @staticmethod
    def ada_ln_modulate(
        x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies adaLN modulation to the input tensor.

        Args:
            x: The input tensor of shape (bsz, seq_len, dim).
            shift: The shift parameter tensor of shape (bsz, 1, dim).
            scale: The scale parameter tensor of shape (bsz, 1, dim).

        Returns:
            The modulated output tensor of shape (bsz, seq_len, dim).
        """
        return x * (1.0 + scale) + shift


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


class DDiTLayer(nn.Module):
    """One layer of DDiT.

    It consists of a multi-head self-attention layer followed by a feedforward layer with adaLN and gating in between.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        d_cond: Optional[int] = None,
        force_flash_attn: bool = False,
    ):
        """
        Initialize the DDiTBlock.

        Args:
            d_model: the dimension of the input.
            nhead: the number of attention heads.
            d_cond: the dimension of the conditioning input.
            mlp_ratio: the ratio of the hidden size of the MLP/feedforward layer to the input size.
            dropout: the dropout rate.
        """
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        d_cond = d_cond or d_model // 2
        self.n_heads = nhead
        self.dim = d_model
        self.norm1 = LayerNormAndScale(d_model, eps=layer_norm_eps)
        self.dropout = dropout
        self.head_dim = d_model // nhead

        # self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim)
        self.rotary_emb = None

        # Single QKV projection
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = LayerNormAndScale(d_model, eps=layer_norm_eps)
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
        self.ada_ln_modulations = AdaLNModulations(d_cond, d_model)
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
        x: torch.Tensor,
        c: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).
            c: the conditioning input of shape (bsz, cond_dim).
            attention_mask: the attention mask of shape (bsz, seq_len), which is True for non-padding tokens.
        """
        if self.rotary_emb is None:
            raise ValueError(
                "RotaryEmbedding is not set. Call set_rotary_emb() to set it."
            )

        # modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_ln_modulations(c)
        )  # shapes: (bsz, 1, dim)

        # Apply adaLN before the attention
        x = AdaLNModulations.ada_ln_modulate(
            self.norm1(x), shift_msa, scale_msa
        )

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
        attn_mask = (
            attention_mask.unsqueeze(-2).unsqueeze(-2)
            if attention_mask is not None
            else None
        )  # shape (bsz, 1, 1, seq_len)

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

        # attn_output = F.scaled_dot_product_attention(
        #    q_rotary,
        #    k_rotary,
        #    v,
        #    attn_mask=attn_mask,
        #    dropout_p=self.dropout if self.training else 0.0,
        # )  # shape (bsz, n_heads, seq_len, head_dim)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], seq_len, self.dim)
        )  # shape (bsz, seq_len, dim)
        x_attn = self.o_proj(attn_output)  # shape (bsz, seq_len, dim)

        # Apply gating and residual connection
        x = add_bias_apply_dropout_scale(
            x_attn,
            bias=None,
            dropout=self.dropout,
            scale=gate_msa,
            residual=x,
            training=self.training,
        )

        # AdaLN -> MLP -> dropout -> scale -> residual
        x = add_bias_apply_dropout_scale(
            self.mlp(
                AdaLNModulations.ada_ln_modulate(
                    self.norm2(x), shift_mlp, scale_mlp
                )
            ),
            bias=None,
            dropout=self.dropout,
            scale=gate_mlp,
            residual=x,
            training=self.training,
        )

        return x

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


class DDiTLayerList(nn.ModuleList):
    """A module list of DDiT blocks that share the rotary cache for the rotary embeddings."""

    def __init__(self, blocks: List[DDiTLayer], rotary_emb: RotaryEmbedding):

        for block in blocks:
            block.set_rotary_emb(rotary_emb)
        super().__init__(blocks)

    @classmethod
    def from_layer(
        cls, layer: DDiTLayer, num_layers: int, rotary_emb: RotaryEmbedding
    ):
        return cls(
            [copy.deepcopy(layer) for _ in range(num_layers)], rotary_emb
        )


class DDitFinalLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_dims: int,
        d_cond: int,
        layer_norm_eps: float = 1e-5,
        use_bias: bool = False,
    ):
        super().__init__()
        self.norm_final = LayerNormAndScale(d_model, eps=layer_norm_eps)
        self.linear = nn.Linear(d_model, out_dims, bias=use_bias)
        with torch.no_grad():
            self.linear.weight.zero_()  # zero init for absorbing diffusion
            # IMPORTANT: zero initialization will not work at all for var len
            # self.linear.weight.data.fill_(1.0)
        self.adaLN_modulation = AdaLNModulations(
            d_cond, d_model, num_modulation_parameters=2
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).
            c: the conditioning input of shape (bsz, cond_dim).
        """
        shift, scale = self.adaLN_modulation(c)
        # region: DEBUG_SPARSE (remove normalization)
        x = self.adaLN_modulation.ada_ln_modulate(
            self.norm_final(x), shift, scale
        )
        # endregion DEBUG_SPARSE
        x = self.linear(x)
        return x


class DDitFinalLayerWithoutNormalization(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_dims: int,
    ):
        super().__init__()
        self.linear = nn.Linear(d_model, out_dims, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()  # zero init for absorbing diffusion

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).
            c: the conditioning input of shape (bsz, cond_dim).
        """
        x = self.linear(x)
        return x


class DDitFinalLayerForClassification(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_dims: int,
        d_cond: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_bias: bool = False,
    ):
        super().__init__()
        self.norm_final = LayerNormAndScale(d_model, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=True),
            nn.Tanh(),
            nn.Linear(2 * d_model, d_model, bias=True),
        )
        self.linear = nn.Linear(d_model, out_dims, bias=use_bias)

        self.adaLN_modulation = AdaLNModulations(
            d_cond, d_model, num_modulation_parameters=3
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).
            c: the conditioning input of shape (bsz, cond_dim).
        """
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)
        # AdaLN -> MLP -> dropout -> scale -> residual
        x = add_bias_apply_dropout_scale(
            self.mlp(
                AdaLNModulations.ada_ln_modulate(
                    self.norm_final(x), shift_mlp, scale_mlp
                )
            ),
            bias=None,
            dropout=self.dropout,
            scale=gate_mlp,
            residual=x,
            training=self.training,
        )

        x = self.linear(x)
        return x
