"""Type definitions for Bd3lm model.

This file defines the data structures used throughout the Bd3lm implementation.
"""

from typing import Optional, Protocol, List, TypedDict
from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


class Bd3lmBatch(TypedDict):
    """A pre-training batch, from DefaultBd3lmCollator.

    Diffusion denoises in place, so target_ids is the clean sequence itself.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): cat(xt, x0), the noisy/clean pair the
            model attends over.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        target_ids (Integer[TT, " batch seq_len"]): the clean sequence (== x0).
    """
    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    target_ids: Integer[TT, " batch seq_len"]


class Bd3lmSeq2SeqBatch(TypedDict):
    """A seq2seq batch, from Bd3lmSeq2SeqCollator.

    The prompt stays clean and only the answer is noised, so target_ids is just the
    clean answer.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): cat(xt, x0), each of them
            prompt + answer.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): Token type ids (not used but kept for interface consistency).
        target_ids (Integer[TT, " batch target_len"]): the clean answer, answer-width
            only - the loss slices the last target_len positions to match it.
    """
    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    token_type_ids: Integer[TT, " batch seq_len"]
    target_ids: Integer[TT, " batch seq_len"]


class Bd3lmLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): per-token NLL over the masked positions only, i.e.
            nlls.sum() / loss_mask.sum(). For a diffusion model this is a variational
            upper bound on the true NLL, so exp(loss) is an upper bound on perplexity.
    """
    loss: Float[TT, ""]


class Bd3lmPredictionDict(TypedDict):
    """Output of the Predictor for Bd3lm.

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


class Bd3lmModel(Protocol):
    """Protocol defining the interface for Bd3lm models.
    """

    def __call__(
        self,
        indices: Integer[TT, " batch two_seq_len"],
        sigma: Optional[Float[TT, " batch"]],
        attention_mask: Optional[Bool[TT, " batch seq_len"]] = None,
        positions: Optional[Integer[TT, " batch seq_len"]] = None,
        **kwargs
    ) -> Float[TT, " batch seq_len vocab_size"]:
        """Forward pass of the model.

        Args:
            indices: cat(xt, x0), so twice the sequence length during training - the
                block-causal mask spans both halves. At sampling time it is just the
                current window.
            sigma: noise level per example. None, or zeroed, when
                algo.time_conditioning is false.
            attention_mask: 1 for non-padding tokens. None when nothing is padded,
                which takes the cheaper rotary path.
            positions: per-row position ids, from cumsum(attention_mask) - 1 so that
                left-padding does not consume position indices. None means plain
                arange.
            **kwargs: sample_mode / store_kv, used by the sampler.

        Returns:
            vocab_logits: shape (batch, seq_len, vocab_size) - the x0 half is dropped
                on the way out during training.
        """
        ...
