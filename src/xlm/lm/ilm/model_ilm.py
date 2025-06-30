from typing import Optional, Tuple

import torch
import torch.nn as nn
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

from xlm.model import Model
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


########################################################
# region: Rotary Transformer


class BaseRotaryTransformerILMModel(torch.nn.Module, Model):
    "Rotary embedding based transformer decoder."

    def __init__(
        self,
        num_embeddings: int,  # vocab plus mask and padding other special tokens
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
        final_layer_without_normalization: bool = True,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.embed_tokens = nn.Embedding(
            num_embeddings, d_model, padding_idx=padding_idx
        )
        self.dim_feedforward = dim_feedforward or 4 * d_model
        encoder_layer = RotaryTransformerLayer(
            d_model,
            nhead,
            self.dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            force_flash_attn=force_flash_attn,
        )
        self.max_length = max_length
        self.encoder = RotaryTransformerLayerList.from_layer(
            encoder_layer,
            num_layers,
            RotaryEmbedding(
                rotary_emb_dim, head_first=True, cache_size=max_length
            ),
        )
        self.output_layer = RotaryTransformerFinalLayer(
            d_model,
            num_embeddings,
            layer_norm_eps,
            use_final_layer_norm=not final_layer_without_normalization,
        )
        self.num_embeddings = num_embeddings

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        token_type_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
    ) -> Tuple[
        Float[TT, " *batch seq_len vocab_size"],
        Float[TT, " *batch max_length"],
    ]:
        """
        Args:
            x_t: The input tokens of shape (*batch, seq_len)
            t: The timesteps of shape (*batch)
            attention_mask: The attention mask of shape (*batch, seq_len), which is True for non-padding tokens.
            positions: The positions of the tokens of shape (*batch, seq_len)
        """
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        x = self.embed_tokens(x_t)  # shape (batch_size, seq_len, d_model)

        for block in self.encoder:
            x = block(x, attention_mask, positions=positions)

        vocab_logits = self.output_layer(
            x,
        )  # shape (batch_size, seq_len, vocab_size)
        return x, vocab_logits

    def get_named_params_for_weight_decay(self):
        # all parameters except biases and layer-norm parameters
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        # biases and layer-norm parameters
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                yield (name, param)


class RotaryTransformerITModel(BaseRotaryTransformerILMModel):
    def forward(  # type: ignore
        self,
        x_t: Integer[TT, " *batch seq_len"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        token_type_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
    ) -> Float[TT, " *batch seq_len vocab_size"]:
        x, vocab_logits = super().forward(
            x_t,
            attention_mask=attention_mask,
            positions=positions,
            token_type_ids=token_type_ids,
        )
        return vocab_logits


class RotaryTransformerILMModel(BaseRotaryTransformerILMModel):
    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        token_type_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
    ) -> Tuple[Float[TT, " *batch seq_len vocab_size"], None]:
        x, vocab_logits = super().forward(
            x_t,
            attention_mask=attention_mask,
            positions=positions,
            token_type_ids=token_type_ids,
        )
        return vocab_logits, None


class RotaryTransformerILMModelWithClassification(
    BaseRotaryTransformerILMModel
):
    "Rotary embedding based transformer decoder."

    def __init__(
        self,
        num_embeddings: int,  # vocab plus mask and padding other special tokens
        d_model: int,
        num_layers: int,
        nhead: int,
        num_classes: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
        final_layer_without_normalization: bool = True,
    ):
        super().__init__(
            num_embeddings,
            d_model,
            num_layers,
            nhead,
            padding_idx=padding_idx,
            mask_idx=mask_idx,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            rotary_emb_dim=rotary_emb_dim,
            max_length=max_length,
            force_flash_attn=force_flash_attn,
            final_layer_without_normalization=final_layer_without_normalization,
        )
        self.num_classes = num_classes

        self.length_output_layer = (
            RotaryTransformerFinalLayerForClassification(
                d_model,
                num_classes,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            )
        )

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        token_type_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
    ) -> Tuple[
        Float[TT, " *batch seq_len vocab_size"],
        Float[TT, " *batch max_length"],
    ]:
        """
        Args:
            x_t: The input tokens of shape (*batch, seq_len)
            t: The timesteps of shape (*batch)
            attention_mask: The attention mask of shape (*batch, seq_len), which is True for non-padding tokens.
            positions: The positions of the tokens of shape (*batch, seq_len)
        """
        x, vocab_logits = super().forward(
            x_t,
            attention_mask,
            positions,
            token_type_ids,
        )  # shape (batch_size, seq_len, d_model), (batch_size, seq_len, vocab_size)
        length_logits = self.length_output_layer(x[:, :1, :]).squeeze(
            1
        )  # shape (batch_size, max_length)
        return vocab_logits, length_logits


class RotaryTransformerILMModelWithStoppingClassification(
    RotaryTransformerILMModelWithClassification
):
    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
        final_layer_without_normalization: bool = True,
    ):
        super().__init__(
            num_embeddings,
            d_model,
            num_layers,
            nhead,
            2,  # num_classes
            padding_idx,
            mask_idx,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            rotary_emb_dim=rotary_emb_dim,
            max_length=max_length,
            force_flash_attn=force_flash_attn,
            final_layer_without_normalization=final_layer_without_normalization,
        )


class RotaryTransformerILMModelWithLengthClassification(
    RotaryTransformerILMModelWithClassification
):
    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        special_final_layer: bool = True,
        force_flash_attn: bool = False,
        num_classes: Optional[int] = None,
    ):
        super().__init__(
            num_embeddings,
            d_model,
            num_layers,
            nhead,
            num_classes if num_classes is not None else max_length,
            padding_idx,
            mask_idx,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            rotary_emb_dim,
            max_length,
            special_final_layer,
            force_flash_attn,
        )

    def get_mean_delta_l(
        self, logits: Float[TT, " *batch num_classes"]
    ) -> Float[TT, " *batch"]:
        delta_l = torch.arange(0, self.num_classes, device=logits.device)
        p = torch.softmax(logits, dim=-1)
        return (p * delta_l).sum(dim=-1)  # shape (*batch)


# endregion: Rotary Transformer
########################################################

########################################################
# region: GPT2


class BaseGPT2ILMModel(GPT, Model):
    def __init__(
        self,
        num_embeddings: int,  # vocab plus mask and padding other special tokens
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_length: int = 1024,
        final_layer_without_normalization: bool = True,
        input_constraint: bool = False,
        keep_type_embeddings: bool = False,
    ):
        gpt_config = GPTConfig(
            block_size=max_length,
            vocab_size=num_embeddings,
            n_layer=num_layers,
            n_head=nhead,
            n_embd=d_model,
            dropout=dropout,
            bias=False,
            padding_idx=padding_idx,
            activation=activation,
            final_layer_without_normalization=final_layer_without_normalization,
            dim_feedforward=(
                dim_feedforward if dim_feedforward is not None else 4 * d_model
            ),
        )
        super().__init__(gpt_config)
        if self.config is None:
            raise ValueError("GPT config is None")
        self.padding_idx = padding_idx
        if input_constraint:
            raise NotImplementedError(
                "Input constraint not implemented for GPT2"
            )
        if keep_type_embeddings:
            raise NotImplementedError(
                "Keep type embeddings not implemented for GPT2"
            )
        self.mask_idx = mask_idx
        self.input_constraint = input_constraint
        self.input_constraint = input_constraint

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        token_type_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
    ) -> Tuple[
        Float[TT, " *batch seq_len vocab_size"],
        Float[TT, " *batch max_length"],
    ]:
        """
        Args:
            x_t: The input tokens of shape (*batch, seq_len)
            t: The timesteps of shape (*batch)
            attention_mask: The attention mask of shape (*batch, seq_len), which is True for non-padding tokens.
            positions: The positions of the tokens of shape (*batch, seq_len)
        """
        device = x_t.device
        idx = x_t
        pos = positions
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)
        b, t = idx.size()  # (batch_size, seq_len)
        if pos is None:
            pos = torch.arange(
                0, t, dtype=torch.long, device=device
            )  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(
            idx
        )  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, attention_mask)
        if not self.config.final_layer_without_normalization:
            x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return x, logits

    def get_named_params_for_weight_decay(self):
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        for name, param in self.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        for name, param in self.named_parameters():
            if param.requires_grad and param.dim() < 2:
                yield (name, param)


class GPT2ILMModel(BaseGPT2ILMModel):
    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        token_type_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
    ) -> Tuple[Float[TT, " *batch seq_len vocab_size"], None]:
        x, vocab_logits = super().forward(
            x_t,
            attention_mask=attention_mask,
            positions=positions,
            token_type_ids=token_type_ids,
        )
        return vocab_logits, None


class GPT2ILMModelWithClassification(BaseGPT2ILMModel):
    def __init__(
        self,
        num_embeddings: int,  # vocab plus mask and padding other special tokens
        d_model: int,
        num_layers: int,
        nhead: int,
        num_classes: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_length: int = 1024,
        final_layer_without_normalization: bool = False,
        keep_type_embeddings: bool = False,  # URGENT: Setting this to True so the running preemptable jobs don't fail
    ):
        super().__init__(
            num_embeddings,
            d_model,
            num_layers,
            nhead,
            padding_idx=padding_idx,
            mask_idx=mask_idx,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_length=max_length,
            final_layer_without_normalization=final_layer_without_normalization,
            keep_type_embeddings=keep_type_embeddings,
        )
        self.num_classes = num_classes

        self.length_output_layer = (
            RotaryTransformerFinalLayerForClassification(
                d_model,
                num_classes,
                dropout=dropout,
            )
        )

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        token_type_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
    ) -> Tuple[
        Float[TT, " *batch seq_len vocab_size"],
        Float[TT, " *batch max_length"],
    ]:
        """
        Args:
            x_t: The input tokens of shape (*batch, seq_len)
            t: The timesteps of shape (*batch)
            attention_mask: The attention mask of shape (*batch, seq_len), which is True for non-padding tokens.
            positions: The positions of the tokens of shape (*batch, seq_len)
        """
        x, vocab_logits = super().forward(
            x_t,
            attention_mask,
            positions,
            token_type_ids,
        )  # shape (batch_size, seq_len, d_model), (batch_size, seq_len, vocab_size)
        length_logits = self.length_output_layer(x[:, :1, :]).squeeze(
            1
        )  # shape (batch_size, max_length)
        return vocab_logits, length_logits


class GPT2ILMModelWithStoppingClassification(GPT2ILMModelWithClassification):
    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_length: int = 1024,
        final_layer_without_normalization: bool = False,
        keep_type_embeddings: bool = False,
    ):
        super().__init__(
            num_embeddings,
            d_model,
            num_layers,
            nhead,
            2,  # num_classes
            padding_idx,
            mask_idx,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_length=max_length,
            final_layer_without_normalization=final_layer_without_normalization,
            keep_type_embeddings=keep_type_embeddings,
        )


class GPT2ILMModelWithLengthClassification(GPT2ILMModelWithClassification):
    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_length: int = 1024,
        special_final_layer: bool = True,
        num_classes: Optional[int] = None,
        keep_type_embeddings: bool = False,
    ):
        super().__init__(
            num_embeddings,
            d_model,
            num_layers,
            nhead,
            num_classes if num_classes is not None else max_length,
            padding_idx,
            mask_idx,
            dim_feedforward,
            dropout,
            activation,
            max_length,
            special_final_layer,
            keep_type_embeddings=keep_type_embeddings,
        )

    def get_mean_delta_l(
        self, logits: Float[TT, " *batch num_classes"]
    ) -> Float[TT, " *batch"]:
        delta_l = torch.arange(0, self.num_classes, device=logits.device)
        p = torch.softmax(logits, dim=-1)
        return (p * delta_l).sum(dim=-1)  # shape (*batch)


# endregion: GPT2
########################################################
