from typing import Optional, Protocol, Tuple, TypedDict, List, Union

from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT

from xlm.datamodule import BaseBatch
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class ZLMBatch(TypedDict):
    """Input to the ZLM.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        target_ids (Integer[TT, " batch seq_len"]): The target ids for language modeling (shifted by 1).
            Positions with -100 are ignored during loss computation (prompt tokens or padding).
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    target_ids: Integer[TT, " batch seq_len"]


class ZLMSeq2SeqPredictionBatch(TypedDict):
    """Input to the ZLM for predicting suffix given the prefix.

    Attributes:
        input_ids (Integer[TT, " batch prefix_seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch prefix_seq_len"]): 1 for tokens that are not padding.
        target_ids (Integer[TT, " batch suffix_seq_len"]): The target ids to the model.
    """

    input_ids: Integer[TT, " batch prefix_seq_len"]
    attention_mask: Integer[TT, " batch prefix_seq_len"]
    target_ids: Integer[TT, " batch suffix_seq_len"]


class ZLMSeq2SeqBatch(TypedDict):
    """Input to the ZLM for sequence-to-sequence training.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model (prompt + target).
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): Token type ids (not used in ZLM but kept for interface consistency).
        target_ids (Integer[TT, " batch seq_len"]): The target ids for language modeling (shifted by 1).
            Positions with -100 are ignored during loss computation (prompt tokens or padding).
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    token_type_ids: Integer[TT, " batch seq_len"]
    target_ids: Integer[TT, " batch seq_len"]


class ZLMUnconditionalPredictionBatch:
    """Input to the ZLM for unconditional generation.

    Attributes:
        input_ids (Integer[TT, " batch 1"]): The input ids to the model (just BOS token).
        attention_mask (Integer[TT, " batch 1"]): 1 for tokens that are not padding.
    """

    input_ids: Integer[TT, " batch 1"]
    attention_mask: Integer[TT, " batch 1"]


class ZLMLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
        batch_loss (Float[TT, " batch"]): Loss value for each example in the batch.
        nlls (Float[TT, " num_tokens"]): The negative log likelihoods of the real predicted tokens (non-pad, and masked in input).
    """

    loss: Float[TT, ""]
    batch_loss: Float[TT, " batch"]
    nlls: Float[TT, " num_tokens"]


TokenLogitsType = Float[TT, " batch seq_len vocab_size"]


class ZLMModel(Protocol):
    def __call__(
        self,
        x_t: Integer[TT, " batch seq_len"],
        attention_mask: Optional[Integer[TT, " batch seq_len seq_len"]] = None,
        positions: Optional[Integer[TT, " batch seq_len"]] = None,
    ) -> TokenLogitsType: ...


class ZLMPredictionDict(TypedDict):
    """Output of the Predictor for ZLM.

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
