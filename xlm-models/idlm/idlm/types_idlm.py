"""Type definitions for Idlm model.

This file defines the data structures used throughout the Idlm implementation.
Based on IDLM v2 types - modified for xLM framework.
"""

from typing import Optional, Protocol, List, TypedDict, Tuple
from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


class IdlmBatch(TypedDict):
    """Input to the Idlm model.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
        target_ids (Optional[Integer[TT, " batch seq_len vocab_size"]]): The target ids to the model. None for prediction.
        n_drops (Optional[Integer[TT, " batch seq_len"]]): 1 for positions where tokens were dropped. None for prediction.
        t (Float[TT, " batch"]): The time step.
        noise_rate (Float[TT, " batch"]): The noise rate.
        total_noise (Float[TT, " batch"]): The total noise value.
        constraint (Optional[Bool[TT, " batch seq_len"]]): 1 for positions out of which there should be no prediction.
            Mostly used during prediction only.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    token_type_ids: Integer[TT, " batch seq_len"]
    target_ids: Optional[Integer[TT, " batch seq_len vocab_size"]]
    cls_position: Optional[Integer[TT, " batch"]]
    n_drops: Optional[Integer[TT, " batch seq_len"]]
    t: Float[TT, " batch"]
    noise_rate: Float[TT, " batch"]
    total_noise: Float[TT, " batch"]
    constraint: Optional[Bool[TT, " batch seq_len"]]


class IdlmSeq2SeqBatch(TypedDict):
    """Input to the Idlm for sequence-to-sequence training.

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


class IdlmLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
        batch_loss (Float[TT, " batch"]): Loss value for each example in the batch.
        per_example_length_loss (Float[TT, " batch"]): Length loss value for each example in the batch.
        per_example_ce (Float[TT, " batch"]): Token prediction loss value for each example in the batch.
        length_logits (Float[TT, " batch max_length"]): The length logits for each example in the batch.
        n_drops (Integer[TT, " batch"]): The number of dropped tokens for each example in the batch.
    """

    loss: Float[TT, ""]
    batch_loss: Float[TT, " batch"]
    per_example_length_loss: Float[TT, " batch"]
    per_example_ce: Float[TT, " batch"]
    length_logits: Float[TT, " batch max_length"]
    n_drops: Integer[TT, " batch"]


class IdlmPredictionDict(TypedDict):
    """Output of the Predictor for Idlm.

    Attributes:
        loss (None): Loss is None for predictions.
        text (List[str]): The batch of generated text without special tokens.
        text_with_spl_tokens (List[str]): The batch of generated text with special tokens.
        ids (Integer[TT, " batch seq_len"]): The batch of generated token_ids.
        attention_mask (Bool[TT, " batch seq_len"]): Attention mask accompanying the generated ids.
        positions (Integer[TT, " batch seq_len"]): The batch of positions of the generated tokens accompanying the ids.
        history (List[List[Tuple[str, float, int]]]): History of generation steps.
    """

    loss: None
    text: List[str]
    text_with_spl_tokens: List[str]
    ids: Integer[TT, " batch seq_len"]
    attention_mask: Bool[TT, " batch seq_len"]
    positions: Integer[TT, " batch seq_len"]
    history: List[List[Tuple[str, float, int]]]


# Type aliases for better readability
TokenLogitsType = Float[TT, " batch seq_len vocab_size"]
LengthLogitsType = Float[TT, " batch max_length"]


class IdlmModel(Protocol):
    """Protocol defining the interface for Idlm models."""

    num_classes: int  # max_length

    def __call__(
        self,
        x_t: Integer[TT, " batch seq_len"],
        t: Float[TT, " batch"],
        non_drop_non_pad: Integer[TT, " batch seq_len"],
        positions: Integer[TT, " batch seq_len"],
        token_type_ids: Optional[Integer[TT, " batch seq_len"]] = None,
        cls_position: Optional[Integer[TT, " batch"]] = None,
    ) -> Tuple[TokenLogitsType, LengthLogitsType]:
        """Forward pass of the model.

        Args:
            x_t: The input tokens of shape (batch, seq_len)
            t: The time step of shape (batch,)
            non_drop_non_pad: Mask indicating non-dropped, non-padded positions
            positions: The positions of the tokens of shape (batch, seq_len)
            token_type_ids: Optional token type ids
        Returns:
            Tuple of (token_logits, length_logits)
        """
        ...

    def get_mean_delta_l(
        self,
        length_logits: LengthLogitsType,
        attention_mask: Optional[Bool[TT, " batch max_length"]] = None,
        temperature: float = 1.0,
    ) -> Float[TT, " batch"]:
        """Get the mean delta length from length logits.

        Args:
            length_logits: The length logits of shape (batch, max_length)
            attention_mask: Optional attention mask
            temperature: Temperature for sampling
        Returns:
            Mean delta length of shape (batch,)
        """
        ...
