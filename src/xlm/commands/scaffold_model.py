#!/usr/bin/env python3
"""
Script to scaffold a new external language model for the XLM framework.

Usage: xlm-scaffold <model_name> [options]

This script creates a complete external model structure with:
- Python package with skeleton implementations
- Configuration files for all necessary components
- Documentation and examples
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Any


def validate_model_name(name: str) -> str:
    """Validate and normalize model name."""
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        raise ValueError(
            f"Model name '{name}' must start with lowercase letter "
            "and contain only lowercase letters, numbers, and underscores"
        )
    return name


def create_template_context(model_name: str) -> Dict[str, Any]:
    """Create template context with all necessary variables."""
    class_name = "".join(word.capitalize() for word in model_name.split("_"))
    return {
        "model_name": model_name,
        "model_name_upper": model_name.upper(),
        "model_class_name": class_name,
        "model_class_prefix": class_name,
    }


def generate_types_file(model_dir: Path, context: Dict[str, Any]) -> None:
    """Generate the types_{model_name}.py file with TypedDict definitions."""
    content = f'''"""Type definitions for {context['model_class_name']} model.

This file defines the data structures used throughout the {context['model_class_name']} implementation.
Based on ARLM types - modify as needed for your specific model.
"""

from typing import Optional, Protocol, List, TypedDict
from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


class {context['model_class_name']}Batch(TypedDict):
    """Input to the {context['model_class_name']} model.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        target_ids (Integer[TT, " batch seq_len"]): The target ids for language modeling (shifted by 1).
            Positions with -100 are ignored during loss computation (prompt tokens or padding).
    """
    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    target_ids: Integer[TT, " batch seq_len"]


class {context['model_class_name']}Seq2SeqBatch(TypedDict):
    """Input to the {context['model_class_name']} for sequence-to-sequence training.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model (prompt + target).
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): Token type ids (not used but kept for interface consistency).
        target_ids (Integer[TT, " batch seq_len"]): The target ids for language modeling (shifted by 1).
            Positions with -100 are ignored during loss computation (prompt tokens or padding).
    """
    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    token_type_ids: Integer[TT, " batch seq_len"]
    target_ids: Integer[TT, " batch seq_len"]


class {context['model_class_name']}LossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
    """
    loss: Float[TT, ""]


class {context['model_class_name']}PredictionDict(TypedDict):
    """Output of the Predictor for {context['model_class_name']}.

    Attributes:
        text (List[str]): The batch of generated text without special tokens.
        text_with_spl_tokens (List[str]): The batch of generated text with special tokens.
        ids (Integer[TT, " batch seq_len"]): The batch of generated token_ids.
        attention_mask (Bool[TT, " batch seq_len"]): Attention mask accompanying the generated ids.
        positions (Integer[TT, " batch seq_len"]): The batch of positions of the generated tokens accompanying the ids.
        time_taken (List[float]): Time taken for each prediction.
        output_start_idx (int): The index of the first output token.
    """
    text: List[str]
    text_with_spl_tokens: List[str]
    ids: Integer[TT, " batch seq_len"]
    attention_mask: Bool[TT, " batch seq_len"]
    positions: Integer[TT, " batch seq_len"]
    time_taken: List[float]
    output_start_idx: int


class {context['model_class_name']}Model(Protocol):
    """Protocol defining the interface for {context['model_class_name']} models."""
    
    def __call__(
        self,
        x_t: Integer[TT, " batch seq_len"],
        attention_mask: Optional[Bool[TT, " batch seq_len seq_len"]] = None,
        positions: Optional[Integer[TT, " batch seq_len"]] = None,
        **kwargs
    ) -> Float[TT, " batch seq_len vocab_size"]:
        """Forward pass of the model.
        
        Args:
            x_t: The input tokens of shape (batch, seq_len)
            attention_mask: The attention mask of shape (batch, seq_len, seq_len) for full attention matrix,
                          or (batch, seq_len) for simple mask. True for non-padding tokens.
            positions: The positions of the tokens of shape (batch, seq_len)
            **kwargs: Additional model-specific arguments
            
        Returns:
            vocab_logits: The vocabulary logits of shape (batch, seq_len, vocab_size)
        """
        ...
'''

    (model_dir / f"types_{context['model_name']}.py").write_text(content)


def generate_model_file(model_dir: Path, context: Dict[str, Any]) -> None:
    """Generate the model_{model_name}.py file with the neural network implementation."""
    content = f'''"""Neural network implementation for {context['model_class_name']} model.

This file contains the main model architecture. Based on ARLM implementation -
modify as needed for your specific model.
"""

from typing import Optional
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT
from xlm.modules.rotary_transformer import (
    RotaryTransformerFinalLayer,
    RotaryTransformerLayer,
    RotaryTransformerLayerList,
    RotaryEmbedding,
)


class RotaryTransformer{context['model_class_name']}Model(torch.nn.Module):
    """Rotary embedding based transformer decoder for auto-regressive language modeling.
    
    This is a working implementation based on ARLM. Modify as needed for your model.
    """

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
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
        final_layer_without_normalization: bool = False,
    ):
        """Initialize the {context['model_class_name']} transformer model.

        Args:
            num_embeddings: Size of the vocabulary.
            d_model: Dimension of the model.
            num_layers: Number of transformer layers.
            nhead: Number of attention heads.
            padding_idx: Index of the padding token.
            dim_feedforward: Dimension of the feedforward network.
            dropout: Dropout rate.
            activation: Activation function.
            layer_norm_eps: Epsilon for layer normalization.
            rotary_emb_dim: Dimension of rotary embeddings.
            max_length: Maximum sequence length.
            force_flash_attn: Whether to force flash attention.
            final_layer_without_normalization: Whether to use final layer without normalization.
        """
        super().__init__()
        self.padding_idx = padding_idx
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
            zero_init=False,
        )

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        attention_mask: Optional[Bool[TT, " *batch seq_len seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        token_type_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
    ) -> Float[TT, " *batch seq_len vocab_size"]:
        """
        Forward pass of the {context['model_class_name']} model.

        Args:
            x_t: The input tokens of shape (*batch, seq_len)
            attention_mask: The attention mask of shape (*batch, seq_len, seq_len) for full attention matrix,
                          or (*batch, seq_len) for simple mask. True for non-padding tokens.
            positions: The positions of the tokens of shape (*batch, seq_len)
            token_type_ids: The token type ids of shape (*batch, seq_len)

        Returns:
            vocab_logits: The vocabulary logits of shape (*batch, seq_len, vocab_size)
        """
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        x = self.embed_tokens(x_t)  # shape (batch_size, seq_len, d_model)

        for block in self.encoder:
            x = block(x, attention_mask, positions=positions)

        vocab_logits = self.output_layer(
            x,
        )  # shape (batch_size, seq_len, vocab_size)
        return vocab_logits

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
'''

    (model_dir / f"model_{context['model_name']}.py").write_text(content)


def generate_loss_file(model_dir: Path, context: Dict[str, Any]) -> None:
    """Generate the loss_{model_name}.py file with loss computation."""
    content = f'''"""Loss function implementation for {context['model_class_name']} model.

This file implements the training loss computation. Modify the loss_fn method
to implement your specific loss computation logic.
"""

from typing import Optional
import torch
import torch.nn.functional as F
from xlm.harness import LossFunction, Harness
from xlm.datamodule import Tokenizer
from .types_{context['model_name']} import {context['model_class_name']}Batch, {context['model_class_name']}LossDict, {context['model_class_name']}Model


class {context['model_class_name']}Loss(LossFunction[{context['model_class_name']}Batch, {context['model_class_name']}LossDict]):
    """Loss function for {context['model_class_name']} model.
    
    TODO: Implement your loss computation logic. Common patterns include:
    - Language modeling loss (cross-entropy on next token prediction)
    - Classification loss 
    - Custom loss functions for specialized tasks
    """

    def __init__(
        self,
        model: Optional[{context['model_class_name']}Model] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """Initialize the loss function.

        Args:
            model: The model instance
            tokenizer: Tokenizer for processing tokens
        """
        self.model = model
        self.tokenizer = tokenizer

    def loss_fn(
        self,
        batch: {context['model_class_name']}Batch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> {context['model_class_name']}LossDict:
        """Compute the causal language modeling loss.

        Args:
            batch: The input batch.
            batch_idx: The batch index.
            dataloader_idx: The dataloader index.
            dataloader_name: The dataloader name.

        Returns:
            Dictionary containing the loss.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_ids = batch["target_ids"]

        assert self.model is not None

        # Create position IDs considering padding on both ends
        # attention_mask has 1 for real tokens, 0 for padding
        positions = attention_mask.cumsum(dim=1) - 1
        positions *= attention_mask  # Zero out positions for padding tokens on the right

        # Create causal attention mask with padding consideration
        _, seq_len = attention_mask.shape
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=attention_mask.device,
            )
        )

        # Expand attention_mask to (batch, seq_len, seq_len) and combine with causal mask
        expanded_attention_mask = attention_mask.unsqueeze(
            1
        )  # (batch, 1, seq_len)
        causal_attention_mask = (
            expanded_attention_mask & causal_mask
        )  # (batch, seq_len, seq_len)

        # Get logits from the model
        logits = self.model(input_ids, causal_attention_mask, positions)

        # For causal LM, we predict the next token
        # Since target_ids are already shifted we don't need to shift again

        # Transpose logits for cross_entropy (expects [N, C, ...] format)
        logits_T = logits.transpose(1, 2)

        # Compute cross-entropy loss (ignores -100 positions)
        ce_loss = torch.nn.functional.cross_entropy(
            logits_T, target_ids, reduction="mean", ignore_index=-100
        )

        return {{
            "loss": ce_loss,
        }}

    def configure(self, pl_module: Harness) -> None:
        """Configure the loss function with the lightning module.
        
        This method is called during setup. Use it for any initialization
        that requires the full lightning module.

        Args:
            pl_module: The lightning module instance
        """
        # TODO: Add any configuration logic here if needed
        # For example:
        # - Set up loss scaling
        # - Initialize auxiliary loss components
        # - Configure metric computation
        pass
'''

    (model_dir / f"loss_{context['model_name']}.py").write_text(content)


def generate_predictor_file(model_dir: Path, context: Dict[str, Any]) -> None:
    """Generate the predictor_{model_name}.py file with inference logic."""
    content = f'''"""Predictor implementation for {context['model_class_name']} model.

This file implements the inference/generation logic.
Based on ARLM implementation - modify as needed.
"""

from typing import List, Dict, Any, Optional, Literal, Tuple
import torch
from jaxtyping import Integer, Bool
from torch import Tensor as TT
from xlm.harness import Predictor
from xlm.datamodule import Tokenizer
from xlm.noise import NoiseSchedule
from .types_{context['model_name']} import {context['model_class_name']}Batch, {context['model_class_name']}PredictionDict, {context['model_class_name']}Model


class {context['model_class_name']}Predictor(Predictor[{context['model_class_name']}Batch, {context['model_class_name']}PredictionDict]):
    """Predictor for {context['model_class_name']} model.
    
    TODO: Implement your prediction/generation logic. Common patterns include:
    - Autoregressive generation (for language models)
    - Parallel generation (for masked language models)
    - Custom generation strategies
    """

    def __init__(
        self,
        model: {context['model_class_name']}Model = None,
        tokenizer: Tokenizer = None,
        noise_schedule: NoiseSchedule = None,
        max_steps: int = 100,
        max_length: int = 512,
        sampling_method: str = "sample_top_p",
        p: float = 0.9,
        top_k: int = 50,
        temperature: float = 1.0,
        **kwargs
    ):
        """Initialize the predictor.

        Args:
            model: The model instance
            tokenizer: Tokenizer for text processing
            noise_schedule: Noise schedule (if applicable)
            max_steps: Maximum number of generation steps
            max_length: Maximum sequence length
            sampling_method: Sampling strategy (sample_top_p, sample_top_k, greedy)
            p: Top-p value for nucleus sampling
            top_k: Top-k value for top-k sampling
            temperature: Temperature for sampling
            **kwargs: Additional generation parameters
        """
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.max_steps = max_steps
        self.max_length = max_length
        self.sampling_method = sampling_method
        self.p = p
        self.top_k = top_k
        self.temperature = temperature

    @torch._dynamo.disable()
    def predict(
        self,
        batch: Dict[str, Any],  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
        max_len: int = 0,
    ) -> {context['model_class_name']}PredictionDict:
        """Generate predictions from the model.

        Args:
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: Dataloader index  
            dataloader_name: Dataloader name
            max_len: Maximum length override

        Returns:
            Dictionary containing generated text and token IDs
        """
        # Record start time
        import time
        start_time = time.time()
        
        # Get batch information
        batch_size = batch["input_ids"].size(0)
        input_length = batch["input_ids"].size(1)
        device = batch["input_ids"].device
        
        # Determine generation length
        generation_steps = max_len if max_len > 0 else self.max_steps
        max_total_length = min(input_length + generation_steps, self.max_length)
        
        # Initialize generation state
        current_ids = batch["input_ids"].clone()
        current_attention_mask = batch["attention_mask"].clone()
        
        # Track positions
        positions = current_attention_mask.cumsum(dim=1) - 1
        positions *= current_attention_mask
        
        # Generate tokens autoregressively
        for step in range(generation_steps):
            if current_ids.size(1) >= max_total_length:
                break
                
            # Create causal attention mask
            seq_len = current_ids.size(1)
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
            )
            expanded_attention_mask = current_attention_mask.unsqueeze(1)
            causal_attention_mask = expanded_attention_mask & causal_mask
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(current_ids, causal_attention_mask, positions)
                
            # Get next token logits  
            next_token_logits = logits[:, -1, :]
            
            # Sample next tokens
            if self.sampling_method == "sample_top_k":
                next_token_ids = self._sample_top_k(next_token_logits)
            elif self.sampling_method == "sample_top_p": 
                next_token_ids = self._sample_top_p(next_token_logits)
            else:  # Default to greedy
                next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append new tokens
            current_ids = torch.cat([current_ids, next_token_ids], dim=1)
            
            # Update attention mask
            new_attention = torch.ones(
                batch_size, 1, dtype=current_attention_mask.dtype, device=device
            )
            current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)
            
            # Update positions
            new_positions = positions.max(dim=1, keepdim=True)[0] + 1
            positions = torch.cat([positions, new_positions], dim=1)
            
            # TODO: Add stopping criteria (EOS detection)
            
        # Decode to text
        generated_text = []
        generated_text_with_spl = []
        for i in range(batch_size):
            tokens = current_ids[i].tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            text_with_spl = self.tokenizer.decode(tokens, skip_special_tokens=False)
            generated_text.append(text)
            generated_text_with_spl.append(text_with_spl)
        
        # Record end time
        end_time = time.time()
        time_taken = [end_time - start_time] * batch_size
        
        return {{
            "text": generated_text,
            "text_with_spl_tokens": generated_text_with_spl,
            "ids": current_ids,
            "attention_mask": current_attention_mask,
            "positions": positions,
            "time_taken": time_taken,
            "output_start_idx": input_length,
        }}

    def _sample_top_k(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k sampling implementation."""
        # TODO: Implement top-k sampling
        # For now, just return greedy
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _sample_top_p(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Top-p (nucleus) sampling implementation."""
        # TODO: Implement top-p sampling
        # For now, just return greedy
        return torch.argmax(logits, dim=-1, keepdim=True)

    def to_dict(self, batch: {context['model_class_name']}Batch, preds: {context['model_class_name']}PredictionDict, **kwargs) -> List[Dict[str, Any]]:
        """Convert predictions to dictionary format for logging.

        Args:
            batch: Original input batch
            preds: Model predictions
            **kwargs: Additional arguments

        Returns:
            List of dictionaries containing prediction results
        """
        # TODO: Customize this based on what you want to log
        
        results = []
        for i in range(len(preds["text"])):
            result = {{
                "generated_text": preds["text"][i],
                "generated_ids": preds["ids"][i].tolist(),
                # TODO: Add any additional fields you want to log
                # For example:
                # "input_text": self.tokenizer.decode(batch["input_ids"][i]),
                # "generation_length": len(preds["ids"][i]) - len(batch["input_ids"][i]),
            }}
            results.append(result)
        
        return results
'''

    (model_dir / f"predictor_{context['model_name']}.py").write_text(content)


def generate_datamodule_file(model_dir: Path, context: Dict[str, Any]) -> None:
    """Generate the datamodule_<model_name>.py file with data processing logic."""
    content = f'''"""Data collation logic for {context['model_class_name']} model.

This file implements the data preprocessing and batching logic.
Based on ARLM implementation - modify as needed.
"""

from typing import List, Dict, Any, Optional, Literal
import torch
from xlm.datamodule import Collator, Tokenizer, Seq2SeqCollatorInput, BaseCollatorInput
from xlm.noise import NoiseSchedule
from xlm.utils.nn import pad_truncate_list
from .types_{context['model_name']} import {context['model_class_name']}Batch, {context['model_class_name']}Seq2SeqBatch


class Default{context['model_class_name']}Collator(Collator):
    """Default collator for {context['model_class_name']} model.
    
    Used for pre-training. Based on ARLM implementation.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        truncate: Literal["max", "block", None] = "block",
        add_eos: bool = False,
    ):
        """Initialize the {context['model_class_name']} collator.

        Args:
            tokenizer: The tokenizer to use.
            block_size: Maximum sequence length.
            noise_schedule: Noise schedule (not used but kept for interface consistency).
            truncate: Truncation strategy.
            add_eos: Whether to add EOS token at the end of the sequence.
        """
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self._vocab_size = len(self.tokenizer)
        self.truncate = truncate
        self.add_eos = add_eos

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not set")
            self._vocab_size = len(self.tokenizer)
        return self._vocab_size

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> {context['model_class_name']}Batch:
        """Collate examples into a batch for {context['model_class_name']} training.

        Args:
            examples: List of examples with input_ids.

        Returns:
            {context['model_class_name']}Batch with input_ids, attention_mask, and target_ids.
        """
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        target_ids: List[List[int]] = []

        # Extract input_ids from examples
        seq_lens = [len(e["input_ids"]) for e in examples]

        # Determine max length based on truncation strategy
        # Account for BOS and EOS tokens that will be added
        tokens_to_add = 1  # BOS token
        if self.add_eos:
            tokens_to_add += 1  # EOS token

        if self.truncate == "max":
            max_len = min(max(seq_lens) + tokens_to_add, self.block_size)
        elif self.truncate == "block":
            max_len = self.block_size
        elif self.truncate is None:
            max_len = max(seq_lens) + tokens_to_add
        else:
            raise ValueError(f"Invalid truncate value: {{self.truncate}}")

        for example in examples:
            # Get the input sequence
            seq = example["input_ids"]

            # Truncate if necessary (account for BOS and EOS tokens)
            if len(seq) > max_len - tokens_to_add:
                seq = seq[: max_len - tokens_to_add]

            # Add BOS token at the beginning
            seq_with_bos = [self.tokenizer.bos_token_id] + seq

            # Add EOS token at the end if requested
            if self.add_eos:
                seq_with_bos = seq_with_bos + [self.tokenizer.eos_token_id]

            # Pad to max_len
            padded_seq = pad_truncate_list(
                seq_with_bos,
                max_len,
                self.tokenizer.pad_token_id,
                pad_left=False,
            )
            input_ids.append(padded_seq)

            # Create attention mask (1 for real tokens including BOS/EOS, 0 for padding)
            mask = [1] * len(seq_with_bos) + [0] * (
                max_len - len(seq_with_bos)
            )
            attention_mask.append(mask)

            # Create target_ids (shifted by 1 for next token prediction)
            # For {context['model_class_name']}, target_ids are the same as input_ids but shifted left by 1
            # Use -100 for padding positions to ignore them during loss computation
            target_seq = seq_with_bos[1:] + [-100]  # Shift left by 1
            # Set padding positions to -100
            for j in range(len(target_seq)):
                if (
                    j < len(mask) - 1 and mask[j + 1] == 0
                ):  # Check if next position is padding
                    target_seq[j] = -100

            target_ids.append(target_seq)

        return {{
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }}


################################################################################
# region: Helper Functions


def prepare_prefix_ids_{context['model_name']}(
    prefix_ids: List[List[int]],
    pad_token_id: int,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    truncate: Literal["max", "block", None] = "block",
    add_bos: Optional[str] = None,
    add_eos: bool = False,
) -> Dict[str, List[List[int]]]:
    """
    Prepare prefix ids for {context['model_class_name']} seq2seq tasks.

    Args:
        prefix_ids: List of prefix token sequences.
        pad_token_id: Padding token ID.
        bos_token_id: BOS token ID.
        eos_token_id: EOS token ID.
        max_seq_len: Maximum sequence length.
        truncate: Truncation strategy.
        add_bos: Where to add BOS token ("input" for prefix, "output" for after prefix, None for no BOS).
        add_eos: Whether to add EOS token at the end of the prefix.

    Returns:
        Dictionary with input_ids and attention_mask as lists.
    """
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []

    # Determine max length
    if truncate in ["max", None]:
        max_len = max(len(_prefix_ids) for _prefix_ids in prefix_ids)
        if truncate == "max" and max_seq_len is not None:
            max_len = max(max_len, max_seq_len)
    elif truncate == "block" and max_seq_len is not None:
        max_len = max_seq_len
    else:
        raise ValueError(f"Invalid truncate, max_seq_len: {{max_seq_len}}")

    assert max_len is not None

    for _prefix_ids in prefix_ids:
        # Add BOS to prefix if requested
        if add_bos == "input" and bos_token_id is not None:
            temp = [bos_token_id] + _prefix_ids
        elif add_bos == "output" and bos_token_id is not None:
            temp = _prefix_ids + [bos_token_id]  # Add BOS to the right
        else:
            temp = _prefix_ids

        # Add EOS token at the end if requested
        if add_eos and eos_token_id is not None:
            temp = temp + [eos_token_id]

        # Pad/truncate
        padded_seq = pad_truncate_list(
            temp, max_len, pad_token_id, pad_left=True
        )
        input_ids.append(padded_seq)

        # Create attention mask (1 for real tokens, 0 for padding on the left)
        mask = [0] * (max_len - len(temp)) + [1] * len(temp)
        attention_mask.append(mask)

    return {{
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }}


def prepare_suffix_ids_{context['model_name']}(
    suffix_ids: List[List[int]],
    pad_token_id: int,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    truncate: Literal["max", "block", None] = "block",
    add_bos: Optional[str] = None,
    add_eos: bool = False,
) -> Dict[str, List[List[int]]]:
    """
    Prepare suffix ids for {context['model_class_name']} seq2seq tasks.

    Args:
        suffix_ids: List of suffix token sequences.
        pad_token_id: Padding token ID.
        bos_token_id: BOS token ID.
        eos_token_id: EOS token ID.
        max_seq_len: Maximum sequence length.
        truncate: Truncation strategy.
        add_bos: Where to add BOS token ("input" for prefix, "output" for after prefix, None for no BOS).
        add_eos: Whether to add EOS token at the end of the suffix.

    Returns:
        Dictionary with input_ids, attention_mask, and target_ids as lists.
    """
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    target_ids: List[List[int]] = []

    # Determine max length
    if truncate in ["max", None]:
        max_len = max(len(_suffix_ids) for _suffix_ids in suffix_ids)
        if truncate == "max" and max_seq_len is not None:
            max_len = max(max_len, max_seq_len)
    elif truncate == "block" and max_seq_len is not None:
        max_len = max_seq_len
    else:
        raise ValueError(f"Invalid truncate, max_seq_len: {{max_seq_len}}")

    assert max_len is not None

    for _suffix_ids in suffix_ids:
        # Add BOS before suffix if requested
        if add_bos == "output" and bos_token_id is not None:
            temp = [bos_token_id] + _suffix_ids
        else:
            temp = _suffix_ids

        # Add EOS token at the end if requested
        if add_eos and eos_token_id is not None:
            temp = temp + [eos_token_id]

        # Pad/truncate
        padded_seq = pad_truncate_list(
            temp, max_len, pad_token_id, pad_left=False
        )
        input_ids.append(padded_seq)

        # Create attention mask
        mask = [1] * len(temp) + [0] * (max_len - len(temp))
        attention_mask.append(mask)

        # Create target_ids (unshifted - will be shifted in collator if needed)
        target_ids.append(padded_seq)

    return {{
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
    }}


################################################################################
# region: Collators


class {context['model_class_name']}Seq2SeqCollator:
    """Seq2seq collator for {context['model_class_name']} model.
    
    Based on ARLM implementation - modify as needed.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        noise_schedule: NoiseSchedule,
        block_size: Optional[int] = None,
        input_block_size: Optional[int] = None,
        add_bos: Optional[str] = None,
        add_eos: bool = False,
        truncate: Literal["max", "block", None] = "block",
    ):
        """Initialize the {context['model_class_name']} sequence-to-sequence collator.

        Args:
            tokenizer: The tokenizer to use.
            noise_schedule: Noise schedule (not used but kept for interface consistency).
            block_size: Maximum sequence length for the target.
            input_block_size: Maximum sequence length for the input.
            add_bos: Where to add BOS token ("input" for prefix, "output" for after prefix, None for no BOS).
            add_eos: Whether to add EOS token at the end of the suffix.
            truncate: Truncation strategy.
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.input_block_size = input_block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.truncate = truncate
        self._vocab_size = (
            len(self.tokenizer) if self.tokenizer is not None else None
        )

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not set")
            self._vocab_size = len(self.tokenizer)
        return self._vocab_size

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> {context['model_class_name']}Seq2SeqBatch:
        """Collate examples into a batch for {context['model_class_name']} sequence-to-sequence training.

        Args:
            examples: List of examples with prompt_ids and input_ids.

        Returns:
            {context['model_class_name']}Seq2SeqBatch with input_ids, attention_mask, target_ids.
        """
        # Prepare prefix (prompt)
        prefix = prepare_prefix_ids_{context['model_name']}(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=self.input_block_size,
            truncate=self.truncate,
            add_bos=self.add_bos,
            add_eos=False,  # No EOS in prefix for seq2seq
        )

        # Prepare suffix (target)
        suffix = prepare_suffix_ids_{context['model_name']}(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=self.block_size,
            truncate=self.truncate,
            add_bos=None,  # BOS through prefix
            add_eos=self.add_eos,
        )

        # Concatenate prefix and suffix as lists
        input_ids = [
            p + s for p, s in zip(prefix["input_ids"], suffix["input_ids"])
        ]
        attention_mask = [
            p + s
            for p, s in zip(prefix["attention_mask"], suffix["attention_mask"])
        ]

        # Create target_ids (shifted by 1 for next token prediction)
        target_ids = []
        for i, (input_seq, mask) in enumerate(zip(input_ids, attention_mask)):
            target_seq = input_seq[1:] + [-100]  # Shift left by 1
            # Set padding positions to -100
            for j in range(len(target_seq)):
                if (
                    j < len(mask) - 1 and mask[j + 1] == 0
                ):  # Check if next position is padding
                    target_seq[j] = -100
            target_ids.append(target_seq)

        return {{
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "token_type_ids": torch.zeros(
                len(input_ids),
                max(len(seq) for seq in input_ids),
                dtype=torch.long,
            ),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }}


class {context['model_class_name']}Seq2SeqPredCollator({context['model_class_name']}Seq2SeqCollator):
    """Drops all the suffix/target tokens and sends them in the target_ids of shape (batch_size, target_seq_len)"""

    def __call__(
        self,
        examples: List[Seq2SeqCollatorInput],
    ) -> {context['model_class_name']}Seq2SeqBatch:
        """Collate examples into a batch for {context['model_class_name']} sequence-to-sequence prediction.

        Args:
            examples: List of examples with prompt_ids and input_ids.

        Returns:
            {context['model_class_name']}Seq2SeqBatch with input_ids, attention_mask, target_ids.
        """
        # For prediction, we only need the prefix (prompt) and the target_ids
        # Prepare prefix (prompt)
        prefix = prepare_prefix_ids_{context['model_name']}(
            [e["prompt_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=self.input_block_size,
            truncate=self.truncate,
            add_bos=self.add_bos,
            add_eos=False,  # No EOS in prefix for seq2seq
        )

        # Prepare target_ids (the full suffix sequence)
        target_ids = prepare_suffix_ids_{context['model_name']}(
            [e["input_ids"] for e in examples],
            self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_seq_len=self.block_size,
            truncate=self.truncate,
            add_bos=None,
            add_eos=self.add_eos,
        )

        # For prediction, input_ids is just the prefix
        input_ids = prefix["input_ids"]
        attention_mask = prefix["attention_mask"]

        # target_ids is the full suffix sequence (not shifted)
        target_ids = target_ids[
            "target_ids"
        ]  # Use unshifted target_ids for prediction

        # Convert to tensors
        return {{
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "token_type_ids": torch.zeros(
                len(input_ids),
                max(len(seq) for seq in input_ids),
                dtype=torch.long,
            ),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }}


# endregion: Collators
################################################################################


################################################################################
# region: Utilities


def _replace_100_with_pad(ids: torch.Tensor, tokenizer: Tokenizer):
    _ids = ids.clone()
    _ids[_ids == -100] = tokenizer.pad_token_id
    return _ids


def print_batch_{context['model_name']}(
    batch: Dict[str, Any],
    split: Literal["train", "val", "test", "predict"],
    tokenizer: Tokenizer,
    dataloader_name: str = "",
):
    """Print batch information for debugging {context['model_class_name']} batches.

    Args:
        batch: The batch to print.
        split: The split name.
        tokenizer: The tokenizer to decode tokens.
        dataloader_name: Name of the dataloader.
    """
    print(
        f"Printing first entries of the tensors in batch for {{split}}/{{dataloader_name}}..."
    )
    print("input tokens:")
    # replace -100 with <pad>
    _input_ids = _replace_100_with_pad(batch["input_ids"][0], tokenizer)
    print(tokenizer.decode(_input_ids))
    print("input_ids:")
    print(batch["input_ids"][0])
    print("attention_mask (int):")
    print(batch["attention_mask"][0].int())
    print("target_ids:")
    print(batch["target_ids"][0])
    print("target tokens:")
    _target_ids = _replace_100_with_pad(batch["target_ids"][0], tokenizer)
    print(tokenizer.decode(_target_ids))


# endregion: Utilities
################################################################################
'''

    (model_dir / f"datamodule_{context['model_name']}.py").write_text(content)


def generate_metrics_file(model_dir: Path, context: Dict[str, Any]) -> None:
    """Generate the metrics_{model_name}.py file with metric computation logic."""
    content = f'''"""Metrics computation for {context['model_class_name']} model.

This file implements metric update functions used by the training framework.
Based on ARLM metrics - modify as needed for your model.
"""

from typing import Any, Dict
import torch


def seq2seq_exact_match_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """
    Args:
        batch: Dict[str, Any]. Should contain the following keys:
            - "target_ids": Integer[TT, " *batch target_seq_len"]
            - "input_ids": Integer[TT, " *batch input_seq_len"]
        loss_dict: Dict[str, Any]. Should contain the following keys:
            - "ids": Integer[TT, " *batch input_seq_len+target_seq_len"]
    Note: We rely on having same number right pads in target and pred, which may not be true for {context['model_class_name']}.
    """
    output_start_idx = loss_dict["output_start_idx"]
    pred = loss_dict["ids"][:, output_start_idx:]
    return {{
        "pred": pred,
        "target": batch["target_ids"],
        "pred_length": None,
        "target_length": None,
    }}


def seq2seq_token_accuracy_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """
    Args:
        batch: Dict[str, Any]. Should contain the following keys:
            - "target_ids": Integer[TT, " *batch target_seq_len"]
            - "input_ids": Integer[TT, " *batch input_seq_len"]
        loss_dict: Dict[str, Any]. Should contain the following keys:
            - "ids": Integer[TT, " *batch input_seq_len+target_seq_len"]
    """
    output_start_idx = loss_dict["output_start_idx"]
    pred = loss_dict["ids"][:, output_start_idx:]
    target = batch["target_ids"]
    pred_mask = torch.ones_like(pred, dtype=torch.bool)
    return {{
        "pred": pred,
        "target": target,
        "pred_mask": pred_mask,
    }}


def mean_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for mean loss metric.

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary containing loss.

    Returns:
        Dictionary with mean loss value.
    """
    return {{
        "value": loss_dict["loss"],
    }}


def perplexity_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for perplexity metric.

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary containing nlls.

    Returns:
        Dictionary with perplexity value.
    """
    # Perplexity is exp(mean(nlls))
    nlls = loss_dict["nlls"]
    perplexity = torch.exp(nlls.mean())
    return {{
        "value": perplexity,
    }}


def token_nll_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for token-level negative log likelihood metric.

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary containing nlls.

    Returns:
        Dictionary with token-level NLL values.
    """
    return {{
        "value": loss_dict["nlls"],
    }}


def sequence_length_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for sequence length metric.

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary.

    Returns:
        Dictionary with sequence length values.
    """
    # Calculate sequence lengths based on attention mask
    attention_mask = batch["attention_mask"]
    seq_lengths = attention_mask.sum(dim=1).float()
    return {{
        "value": seq_lengths,
    }}


def valid_tokens_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Update function for valid tokens count metric.

    Args:
        batch: Input batch.
        loss_dict: Loss dictionary.

    Returns:
        Dictionary with valid tokens count.
    """
    # Count tokens that are not padding (valid tokens)
    attention_mask = batch["attention_mask"]
    target_ids = batch["target_ids"]

    # Valid tokens are those that are not padding and not -100 (ignored tokens)
    valid_tokens = attention_mask & (target_ids != -100)
    valid_token_counts = valid_tokens.sum(dim=1).float()

    return {{
        "value": valid_token_counts,
    }}
'''

    (model_dir / f"metrics_{context['model_name']}.py").write_text(content)


def generate_init_file(
    model_dir: Path, context: Dict[str, Any], is_core: bool = False
) -> None:
    """Generate the __init__.py file."""
    if is_core:
        content = f'''"""
{context['model_class_name']} - Core Language Model for XLM Framework

This module implements the {context['model_class_name']} model with all necessary components:
- Model architecture (model_{context['model_name']}.py)
- Loss function (loss_{context['model_name']}.py) 
- Predictor for inference (predictor_{context['model_name']}.py)
- Data module (datamodule_{context['model_name']}.py)
- Metrics computation (metrics_{context['model_name']}.py)
- Type definitions (types_{context['model_name']}.py)

This is a core model that is part of the XLM library.
"""

from .model_{context['model_name']} import RotaryTransformer{context['model_class_name']}Model
from .loss_{context['model_name']} import {context['model_class_name']}Loss
from .predictor_{context['model_name']} import {context['model_class_name']}Predictor
from .datamodule_{context['model_name']} import Default{context['model_class_name']}Collator, {context['model_class_name']}Seq2SeqCollator
from .types_{context['model_name']} import (
    {context['model_class_name']}Batch,
    {context['model_class_name']}Seq2SeqBatch, 
    {context['model_class_name']}LossDict,
    {context['model_class_name']}PredictionDict,
)

__all__ = [
    "RotaryTransformer{context['model_class_name']}Model",
    "{context['model_class_name']}Loss", 
    "{context['model_class_name']}Predictor",
    "Default{context['model_class_name']}Collator",
    "{context['model_class_name']}Seq2SeqCollator",
    "{context['model_class_name']}Batch",
    "{context['model_class_name']}Seq2SeqBatch",
    "{context['model_class_name']}LossDict",
    "{context['model_class_name']}PredictionDict",
]
'''
    else:
        content = f'''"""
{context['model_class_name']} - External Language Model for XLM Framework

This package implements the {context['model_class_name']} model with all necessary components:
- Model architecture (model_{context['model_name']}.py)
- Loss function (loss_{context['model_name']}.py) 
- Predictor for inference (predictor_{context['model_name']}.py)
- Data module (datamodule_{context['model_name']}.py)
- Metrics computation (metrics_{context['model_name']}.py)
- Type definitions (types_{context['model_name']}.py)

To use this model:
1. Add '{context['model_name']}' to your .xlm_models file
2. Use model_type={context['model_name']} and model={context['model_name']} in your config
"""

from .model_{context['model_name']} import RotaryTransformer{context['model_class_name']}Model
from .loss_{context['model_name']} import {context['model_class_name']}Loss
from .predictor_{context['model_name']} import {context['model_class_name']}Predictor
from .datamodule_{context['model_name']} import Default{context['model_class_name']}Collator, {context['model_class_name']}Seq2SeqCollator
from .types_{context['model_name']} import (
    {context['model_class_name']}Batch,
    {context['model_class_name']}Seq2SeqBatch, 
    {context['model_class_name']}LossDict,
    {context['model_class_name']}PredictionDict,
)

__all__ = [
    "RotaryTransformer{context['model_class_name']}Model",
    "{context['model_class_name']}Loss", 
    "{context['model_class_name']}Predictor",
    "Default{context['model_class_name']}Collator",
    "{context['model_class_name']}Seq2SeqCollator",
    "{context['model_class_name']}Batch",
    "{context['model_class_name']}Seq2SeqBatch",
    "{context['model_class_name']}LossDict",
    "{context['model_class_name']}PredictionDict",
]
'''

    (model_dir / "__init__.py").write_text(content)


def generate_config_files(
    config_dir: Path, context: Dict[str, Any], is_core: bool = False
) -> None:
    """Generate all configuration files."""
    # Model config
    if is_core:
        model_target = f"xlm.lm.{context['model_name']}.model_{context['model_name']}.RotaryTransformer{context['model_class_name']}Model"
        loss_target = f"xlm.lm.{context['model_name']}.loss_{context['model_name']}.{context['model_class_name']}Loss"
        predictor_target = f"xlm.lm.{context['model_name']}.predictor_{context['model_name']}.{context['model_class_name']}Predictor"
        collator_target = f"xlm.lm.{context['model_name']}.datamodule_{context['model_name']}.Default{context['model_class_name']}Collator"
        seq2seq_collator_target = f"xlm.lm.{context['model_name']}.datamodule_{context['model_name']}.{context['model_class_name']}Seq2SeqCollator"
        seq2seq_pred_collator_target = f"xlm.lm.{context['model_name']}.datamodule_{context['model_name']}.{context['model_class_name']}Seq2SeqPredCollator"
        print_batch_fn = f"xlm.lm.{context['model_name']}.datamodule_{context['model_name']}.print_batch_{context['model_name']}"
        metrics_prefix = (
            f"xlm.lm.{context['model_name']}.metrics_{context['model_name']}"
        )
    else:
        model_target = f"{context['model_name']}.model_{context['model_name']}.RotaryTransformer{context['model_class_name']}Model"
        loss_target = f"{context['model_name']}.loss_{context['model_name']}.{context['model_class_name']}Loss"
        predictor_target = f"{context['model_name']}.predictor_{context['model_name']}.{context['model_class_name']}Predictor"
        collator_target = f"{context['model_name']}.datamodule_{context['model_name']}.Default{context['model_class_name']}Collator"
        seq2seq_collator_target = f"{context['model_name']}.datamodule_{context['model_name']}.{context['model_class_name']}Seq2SeqCollator"
        seq2seq_pred_collator_target = f"{context['model_name']}.datamodule_{context['model_name']}.{context['model_class_name']}Seq2SeqPredCollator"
        print_batch_fn = f"{context['model_name']}.datamodule_{context['model_name']}.print_batch_{context['model_name']}"
        metrics_prefix = (
            f"{context['model_name']}.metrics_{context['model_name']}"
        )

    model_config = f"""# @package _global_

model:
  _target_: {model_target}
  num_embeddings: ${{tokenizer:full_vocab_size}}
  d_model: 768
  num_layers: 12
  nhead: 12
  padding_idx: ${{tokenizer:pad_token_id}}
  dim_feedforward: ${{eval:${{.d_model}}*4}}
  dropout: 0.1
  activation: relu
  layer_norm_eps: 1e-5
  rotary_emb_dim: 64
  max_length: ${{predictor.max_length}}
  force_flash_attn: false
  final_layer_without_normalization: false

tags:
  model: {context['model_name']}
"""
    (config_dir / "model" / f"{context['model_name']}.yaml").write_text(
        model_config
    )

    # Model type config
    model_type_config = f"""# @package _global_

defaults:
  - /metrics@reported_metrics.train.lm.accumulated_loss: accumulated_loss
  - /metrics@reported_metrics.val.lm.accumulated_loss: accumulated_loss
  - /metrics@reported_metrics.test.lm.accumulated_loss: accumulated_loss
  - /metrics@reported_metrics.val.prediction.exact_match: seq2seq_exact_match
  - /metrics@reported_metrics.test.prediction.exact_match: seq2seq_exact_match
  - /metrics@reported_metrics.val.prediction.token_accuracy: seq2seq_token_accuracy
  - /metrics@reported_metrics.test.prediction.token_accuracy: seq2seq_token_accuracy

lightning_module:
  _target_: xlm.harness.Harness

loss:
  _target_: {loss_target}

predictor:
  _target_: {predictor_target}
  tokenizer: ${{lightning_module:tokenizer}}
  noise_schedule: ${{lightning_module:noise_schedule}}
  max_steps: ${{block_size}}
  max_length: ${{eval:${{block_size}}+${{oc.select:input_block_size,0}}}}
  sampling_method: sample_top_p
  p: 0.5

diagnostic_metrics: null

reported_metrics:
  train:
    lm:
      accumulated_loss:
        prefix: train/lm
        update_fn: {metrics_prefix}.mean_metric_update_fn
  val:
    lm:
      accumulated_loss:
        prefix: val/lm
        update_fn: {metrics_prefix}.mean_metric_update_fn
    prediction:
      exact_match:
        prefix: val/prediction
        update_fn: {metrics_prefix}.seq2seq_exact_match_update_fn
      token_accuracy:
        prefix: val/prediction
        update_fn: {metrics_prefix}.seq2seq_token_accuracy_update_fn
  test:
    lm:
      accumulated_loss:
        prefix: test/lm
        update_fn: {metrics_prefix}.mean_metric_update_fn
    prediction:
      exact_match:
        prefix: test/prediction
        update_fn: {metrics_prefix}.seq2seq_exact_match_update_fn
      token_accuracy:
        prefix: test/prediction
        update_fn: {metrics_prefix}.seq2seq_token_accuracy_update_fn

tags:
  model_type: {context['model_name']}
"""
    (config_dir / "model_type" / f"{context['model_name']}.yaml").write_text(
        model_type_config
    )

    # Collator configs
    default_collator_config = f"""_target_: {collator_target}
block_size: ${{block_size}}
tokenizer: ${{global_components:tokenizer}}
noise_schedule: ${{global_components:noise_schedule}}
"""
    (
        config_dir / "collator" / f"default_{context['model_name']}.yaml"
    ).write_text(default_collator_config)

    seq2seq_collator_config = f"""_target_: {seq2seq_collator_target}
input_block_size: ${{oc.select:input_block_size,null}}
block_size: ${{block_size}}
tokenizer: ${{global_components:tokenizer}}
noise_schedule: ${{global_components:noise_schedule}}
add_bos: output
add_eos: true
"""
    (
        config_dir / "collator" / f"seq2seq_{context['model_name']}.yaml"
    ).write_text(seq2seq_collator_config)

    # Datamodule configs
    (config_dir / "datamodule").mkdir(exist_ok=True)

    # Base star datamodule config
    star_datamodule_config = f"""# @package _global_
defaults:
  - default
  - /collator@datamodule.dataset_managers.train.lm.collator: seq2seq_{context['model_name']}
  - /collator@datamodule.dataset_managers.val.lm.collator: seq2seq_{context['model_name']}
  - /collator@datamodule.dataset_managers.test.lm.collator: seq2seq_{context['model_name']}
  - /collator@datamodule.dataset_managers.val.prediction.collator: seq2seq_pred_{context['model_name']}
  - /collator@datamodule.dataset_managers.test.prediction.collator: seq2seq_pred_{context['model_name']}
  - /collator@datamodule.dataset_managers.predict.prediction.collator: seq2seq_pred_{context['model_name']}

datamodule:
  print_batch_fn: {print_batch_fn}

tags:
  dataset: ???
"""
    (
        config_dir / "datamodule" / f"star_{context['model_name']}.yaml"
    ).write_text(star_datamodule_config)

    # Star easy datamodule config
    star_easy_datamodule_config = f"""# @package _global_
defaults:
  - star_{context['model_name']}
  - /datasets@datamodule.dataset_managers.train.lm: star_easy_train
  - /datasets@datamodule.dataset_managers.val.lm: star_easy_val
  - /datasets@datamodule.dataset_managers.val.prediction: star_easy_val_pred
  - /datasets@datamodule.dataset_managers.test.lm: star_easy_test
  - /datasets@datamodule.dataset_managers.test.prediction: star_easy_test_pred
  - /datasets@datamodule.dataset_managers.predict.prediction: star_easy_test_pred

tags:
  dataset: star_easy
"""
    (
        config_dir / "datamodule" / f"star_easy_{context['model_name']}.yaml"
    ).write_text(star_easy_datamodule_config)

    # Seq2seq pred collator config
    seq2seq_pred_collator_config = f"""_target_: {seq2seq_pred_collator_target}
input_block_size: ${{oc.select:input_block_size,null}}
block_size: ${{block_size}}
tokenizer: ${{global_components:tokenizer}}
noise_schedule: ${{global_components:noise_schedule}}
add_bos: output
add_eos: true
"""
    (
        config_dir / "collator" / f"seq2seq_pred_{context['model_name']}.yaml"
    ).write_text(seq2seq_pred_collator_config)

    # Experiment config - using star_easy pattern
    experiment_config = f"""# @package _global_
defaults:
  - override /datamodule: star_easy_{context['model_name']}
  - override /noise_schedule: dummy # default
  - override /model_type: {context['model_name']}
  - override /model: {context['model_name']}

per_device_batch_size: 64
global_batch_size: 64
input_block_size: 28
block_size: 14

datamodule:
  print_batch_fn: {print_batch_fn}

global_components:
  tokenizer:
    _target_: xlm.datamodule.SimpleSpaceTokenizer.for_numbers
    vocab_size: 20 # 20 (easy,medium), 56 (hard)

trainer:
  max_steps: 80000 
  val_check_interval: null
  num_sanity_val_steps: 3
  check_val_every_n_epoch: 2

log_predictions:
  _target_: xlm.log_predictions.LogPredictions
  fields_to_keep_in_output:
    - text
    - truth
  inject_target: target_ids
  writers:
    - file
    - logger

callbacks:
  checkpoint_monitor:
    monitor: val/lm/accumulated_loss

optimizer:
  lr: 0.0001

lr_scheduler:
  name: "constant" # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L435
  num_warmup_steps: 500
  num_training_steps: ${{trainer.max_steps}}
  monitor: "train/loss"

predictor:
  sampling_method: sample_top_k
  top: 1
"""
    (
        config_dir / "experiment" / f"star_easy_{context['model_name']}.yaml"
    ).write_text(experiment_config)


def generate_setup_file(model_dir: Path,context: Dict[str, Any]) -> None:
    """Generate setup.py for the external model."""
    content = f"""from setuptools import setup, find_packages

setup(
    name="{context['model_name']}",
    version="0.1.0", 
    description="{context['model_class_name']} - External Language Model for XLM framework",
    packages=find_packages(),
    install_requires=[
        "xlm",  # Main XLM package dependency. Add any other dependencies your model needs
    ],
    package_data={{
        "{context['model_name']}": ["configs/**/*.yaml"],
    }},
    include_package_data=True,
    python_requires=">=3.11",
    author="Your Name",
    author_email="your.email@example.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers", 
        "Programming Language :: Python :: 3.11",
    ],
)
"""
    (model_dir.parent / "setup.py").write_text(content)


def generate_documentation(model_dir: Path, context: Dict[str, Any]) -> None:
    """Generate README and documentation."""
    readme_content = f"""# {context['model_class_name']} - External Language Model for XLM

This is an external language model implementation for the XLM framework.

## Structure

- `{context['model_name']}/` - Python package containing the model implementation
- `configs/` - Hydra configuration files for the model
- `setup.py` - Package installation script

## Implementation Status

This is a scaffolded implementation. You need to complete the following:

### Required Implementations

1. **Model Architecture** (`{context['model_name']}/model_{context['model_name']}.py`):
   - [ ] Implement the neural network architecture in `RotaryTransformer{context['model_class_name']}Model.forward()`
   - [ ] Add any custom layers or components your model needs
   - [ ] Configure model parameters in the `__init__` method

2. **Loss Function** (`{context['model_name']}/loss_{context['model_name']}.py`):
   - [ ] Implement loss computation in `{context['model_class_name']}Loss.loss_fn()`
   - [ ] Add any additional metrics you want to track
   - [ ] Configure loss-specific parameters

3. **Predictor** (`{context['model_name']}/predictor_{context['model_name']}.py`):
   - [ ] Implement generation logic in `{context['model_class_name']}Predictor.predict()`
   - [ ] Implement sampling methods (`_sample_top_k`, `_sample_top_p`)
   - [ ] Add stopping criteria and post-processing

4. **Data Module** (`{context['model_name']}/datamodule_{context['model_name']}.py`):
   - [ ] Implement data preprocessing in `Default{context['model_class_name']}Collator.__call__()`
   - [ ] Implement seq2seq preprocessing in `{context['model_class_name']}Seq2SeqCollator.__call__()`
   - [ ] Add any task-specific data transformations

5. **Metrics** (`{context['model_name']}/metrics_{context['model_name']}.py`):
   - [ ] Customize metric update functions for your model
   - [ ] Add any model-specific metrics

6. **Configuration** (`configs/`):
   - [ ] Adjust model parameters in `configs/model/{context['model_name']}.yaml`
   - [ ] Configure training settings in `configs/experiment/{context['model_name']}_debug.yaml`
   - [ ] Create datamodule configs if needed

## Usage

1. **Add to XLM models list**:
   ```bash
   echo "{context['model_name']}" >> .xlm_models
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

3. **Run a debug training**:
   ```bash
   xlm job_type=train \\
     job_name={context['model_name']}_debug \\
     experiment={context['model_name']}_debug \\
     debug=overfit
   ```

## Development Tips

- Start with the model architecture - this is the core component
- Test each component independently before integration
- Use the debug configs for fast iteration
- Check the XLM documentation for examples from existing models (ARLM, ILM, MLM)
- Add print statements and logging to understand data flow

## Configuration

The model can be configured through Hydra configs:

- `model={context['model_name']}` - Use this model architecture
- `model_type={context['model_name']}` - Use this model's loss/predictor/metrics
- `experiment={context['model_name']}_debug` - Use the debug experiment configuration

## File Structure

```
{context['model_name']}/
├── {context['model_name']}/           # Python package
│   ├── __init__.py                   # Package exports
│   ├── types_{context['model_name']}.py              # Type definitions
│   ├── model_{context['model_name']}.py              # Neural network architecture
│   ├── loss_{context['model_name']}.py               # Loss function
│   ├── predictor_{context['model_name']}.py          # Inference logic
│   ├── datamodule_{context['model_name']}.py         # Data preprocessing
│   └── metrics_{context['model_name']}.py            # Metrics computation
├── configs/                 # Hydra configurations
│   ├── model/
│   ├── model_type/
│   ├── collator/
│   └── experiment/
├── setup.py                 # Package installation
└── README.md               # This file
```

## Next Steps

1. Implement the model architecture based on your research
2. Test with a simple dataset using the debug configuration
3. Iterate on the loss function and training settings
4. Add custom metrics and evaluation logic
5. Scale up to full datasets and experiments

Good luck with your model development!
"""
    (model_dir.parent / "README.md").write_text(readme_content)


def update_xlm_models_file(
    model_name: str, xlm_models_path: Path = Path(".xlm_models")
) -> None:
    """Add the new model to the .xlm_models file."""
    # Read existing models
    existing_models = []
    if xlm_models_path.exists():
        with open(xlm_models_path, "r") as f:
            existing_models = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]

    # Add new model if not already present
    if model_name not in existing_models:
        with open(xlm_models_path, "a") as f:
            if existing_models:  # Add newline if file is not empty
                f.write("\n")
            f.write(f"{model_name}\n")
        print(f"Added '{model_name}' to {xlm_models_path}")
    else:
        print(f"Model '{model_name}' already exists in {xlm_models_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold a new external language model for XLM framework"
    )
    parser.add_argument(
        "model_name", help="Name of the new model (e.g., 'my_transformer')"
    )
    parser.add_argument(
        "--core",
        action="store_true",
        help="Generate a core model that gets added to the XLM library (deprecated - external models are recommended)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without creating files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to create the model in (default: Current Directory)",
    )
    parser.add_argument(
        "--no-xlm-models",
        action="store_true",
        help="Don't update .xlm_models file (for external models only)",
    )

    args = parser.parse_args()

    try:
        model_name = validate_model_name(args.model_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    context = create_template_context(model_name)

    if args.core:
        # Generate core model in temp directory first
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model_dir = Path(temp_dir) / model_name
            temp_model_dir.mkdir(exist_ok=True)
            (temp_model_dir / model_name).mkdir(exist_ok=True)
            (temp_model_dir / "configs" / "model").mkdir(
                parents=True, exist_ok=True
            )
            (temp_model_dir / "configs" / "model_type").mkdir(exist_ok=True)
            (temp_model_dir / "configs" / "collator").mkdir(exist_ok=True)
            (temp_model_dir / "configs" / "experiment").mkdir(exist_ok=True)
            (temp_model_dir / "configs" / "datamodule").mkdir(exist_ok=True)

            print(f"Scaffolding core model: {model_name}")
            print(f"Temporary directory: {temp_model_dir}")
            if args.dry_run:
                print("DRY RUN - no files will be created")
                print("\n📁 Files that would be created:")

                # Python files in src/xlm/lm/
                core_model_dir = Path("src/xlm/lm") / model_name
                print(f"\n📂 Python files in {core_model_dir}:")
                python_files = [
                    "__init__.py",
                    f"types_{model_name}.py",
                    f"model_{model_name}.py",
                    f"loss_{model_name}.py",
                    f"predictor_{model_name}.py",
                    f"datamodule_{model_name}.py",
                    f"metrics_{model_name}.py",
                ]
                for py_file in python_files:
                    print(f"  📄 {core_model_dir / py_file}")

                # Config files in src/xlm/configs/lightning_train/
                config_base = Path("src/xlm/configs/lightning_train")
                print(f"\n📂 Config files in {config_base}:")

                # Model configs
                print(f"  📂 {config_base / 'model'}:")
                print(f"    📄 {config_base / 'model' / f'{model_name}.yaml'}")

                # Model type configs
                print(f"  📂 {config_base / 'model_type'}:")
                print(
                    f"    📄 {config_base / 'model_type' / f'{model_name}.yaml'}"
                )

                # Collator configs
                print(f"  📂 {config_base / 'collator'}:")
                print(
                    f"    📄 {config_base / 'collator' / f'default_{model_name}.yaml'}"
                )
                print(
                    f"    📄 {config_base / 'collator' / f'seq2seq_{model_name}.yaml'}"
                )
                print(
                    f"    📄 {config_base / 'collator' / f'seq2seq_pred_{model_name}.yaml'}"
                )

                # Datamodule configs
                print(f"  📂 {config_base / 'datamodule'}:")
                print(
                    f"    📄 {config_base / 'datamodule' / f'star_{model_name}.yaml'}"
                )
                print(
                    f"    📄 {config_base / 'datamodule' / f'star_easy_{model_name}.yaml'}"
                )

                # Experiment configs
                print(f"  📂 {config_base / 'experiment'}:")
                print(
                    f"    📄 {config_base / 'experiment' / f'star_easy_{model_name}.yaml'}"
                )

                print(
                    f"\n📝 Total: {len(python_files) + 7} files would be created"
                )
                return

            # Generate Python files
            print("Generating Python package...")
            generate_init_file(
                temp_model_dir / model_name, context, is_core=True
            )
            generate_types_file(temp_model_dir / model_name, context)
            generate_model_file(temp_model_dir / model_name, context)
            generate_loss_file(temp_model_dir / model_name, context)
            generate_predictor_file(temp_model_dir / model_name, context)
            generate_datamodule_file(temp_model_dir / model_name, context)
            generate_metrics_file(temp_model_dir / model_name, context)

            # Generate config files
            print("Generating configuration files...")
            generate_config_files(
                temp_model_dir / "configs", context, is_core=True
            )

            if not args.dry_run:
                # Move Python files to src/xlm/lm/
                core_model_dir = Path("src/xlm/lm") / model_name
                core_model_dir.mkdir(exist_ok=True)

                print(f"Moving Python files to {core_model_dir}...")
                for py_file in (temp_model_dir / model_name).glob("*.py"):
                    shutil.copy2(py_file, core_model_dir / py_file.name)

                # Move config files to src/xlm/configs/
                print("Moving config files to src/xlm/configs/...")
                for config_type in [
                    "model",
                    "model_type",
                    "collator",
                    "datamodule",
                ]:
                    config_src_dir = temp_model_dir / "configs" / config_type
                    config_dst_dir = (
                        Path("src/xlm/configs/lightning_train") / config_type
                    )
                    config_dst_dir.mkdir(exist_ok=True)

                    for config_file in config_src_dir.glob("*.yaml"):
                        shutil.copy2(
                            config_file, config_dst_dir / config_file.name
                        )

                # Move experiment configs
                exp_src_dir = temp_model_dir / "configs" / "experiment"
                exp_dst_dir = Path(
                    "src/xlm/configs/lightning_train/experiment"
                )
                exp_dst_dir.mkdir(exist_ok=True)

                for exp_file in exp_src_dir.glob("*.yaml"):
                    shutil.copy2(exp_file, exp_dst_dir / exp_file.name)

        print(f"✅ Core model '{model_name}' scaffolded successfully!")
        print(f"📁 Python files created in: src/xlm/lm/{model_name}/")
        print("📁 Config files created in: src/xlm/configs/lightning_train/")
        print("📝 Next steps:")
        print(
            f"1. Review and implement the model architecture in src/xlm/lm/{model_name}/model_{model_name}.py"
        )
        print(
            f"2. Review and implement the loss function in src/xlm/lm/{model_name}/loss_{model_name}.py"
        )
        print(
            f"3. Test with: xlm job_type=train experiment=star_easy_{model_name} debug=overfit"
        )

    else:
        # Generate external model
        model_dir = args.output_dir / model_name

        print(f"Scaffolding external model: {model_name}")
        print(f"Output directory: {model_dir}")
        if args.dry_run:
            print("DRY RUN - no files will be created")
            print("\n📁 Files that would be created:")

            # Python files in model_dir/model_name/
            python_dir = model_dir / model_name
            print(f"\n📂 Python files in {python_dir}:")
            python_files = [
                "__init__.py",
                f"types_{model_name}.py",
                f"model_{model_name}.py",
                f"loss_{model_name}.py",
                f"predictor_{model_name}.py",
                f"datamodule_{model_name}.py",
                f"metrics_{model_name}.py",
            ]
            for py_file in python_files:
                print(f"  📄 {python_dir / py_file}")

            # Config files in model_dir/configs/
            config_base = model_dir / "configs"
            print(f"\n📂 Config files in {config_base}:")

            # Model configs
            print(f"  📂 {config_base / 'model'}:")
            print(f"    📄 {config_base / 'model' / f'{model_name}.yaml'}")

            # Model type configs
            print(f"  📂 {config_base / 'model_type'}:")
            print(
                f"    📄 {config_base / 'model_type' / f'{model_name}.yaml'}"
            )

            # Collator configs
            print(f"  📂 {config_base / 'collator'}:")
            print(
                f"    📄 {config_base / 'collator' / f'default_{model_name}.yaml'}"
            )
            print(
                f"    📄 {config_base / 'collator' / f'seq2seq_{model_name}.yaml'}"
            )
            print(
                f"    📄 {config_base / 'collator' / f'seq2seq_pred_{model_name}.yaml'}"
            )

            # Datamodule configs
            print(f"  📂 {config_base / 'datamodule'}:")
            print(
                f"    📄 {config_base / 'datamodule' / f'star_{model_name}.yaml'}"
            )
            print(
                f"    📄 {config_base / 'datamodule' / f'star_easy_{model_name}.yaml'}"
            )

            # Experiment configs
            print(f"  📂 {config_base / 'experiment'}:")
            print(
                f"    📄 {config_base / 'experiment' / f'star_easy_{model_name}.yaml'}"
            )

            # Package files
            print(f"\n📂 Package files in {model_dir}:")
            package_files = ["setup.py", "README.md"]
            for pkg_file in package_files:
                print(f"  📄 {model_dir / pkg_file}")

            print(
                f"\n📝 Total: {len(python_files) + 7 + len(package_files)} files would be created"
            )
            return

        # Create directory structure
        model_dir.mkdir(exist_ok=True)
        (model_dir / "configs" / "model").mkdir(parents=True, exist_ok=True)
        (model_dir / "configs" / "model_type").mkdir(exist_ok=True)
        (model_dir / "configs" / "collator").mkdir(exist_ok=True)
        (model_dir / "configs" / "experiment").mkdir(exist_ok=True)

        # Generate Python files
        print("Generating Python package...")
        generate_init_file(model_dir, context)
        generate_types_file(model_dir, context)
        generate_model_file(model_dir, context)
        generate_loss_file(model_dir, context)
        generate_predictor_file(model_dir, context)
        generate_datamodule_file(model_dir, context)
        generate_metrics_file(model_dir, context)

        # Generate config files
        print("Generating configuration files...")
        generate_config_files(model_dir / "configs", context)

        # Generate package files
        print("Generating package files...")
        generate_setup_file(model_dir,context)
        generate_documentation(model_dir, context)

        # Update .xlm_models file
        if not args.no_xlm_models:
            update_xlm_models_file(model_name)

        print(f"✅ External model '{model_name}' scaffolded successfully!")
        print(f"📁 Created files in: {model_dir}")
        print("📝 Next steps:")
        print(f"1. cd {model_dir}")
        print("2. Read README.md for implementation guidance")
        print(
            f"3. Implement the model architecture in {model_name}/model_{model_name}.py"
        )
        print(
            f"4. Implement the loss function in {model_name}/loss_{model_name}.py"
        )
        print(
            f"5. Test with: xlm job_type=train experiment=star_easy_{model_name} debug=overfit"
        )


if __name__ == "__main__":
    main()
