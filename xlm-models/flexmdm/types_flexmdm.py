from typing import Optional, Protocol, Tuple, TypedDict, List, Union

from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class FlexMDMBatch(TypedDict):
    """Input to the MLM.
    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        target_ids (Optional[Integer[TT, " batch seq_len"]]): The target ids to the model.
    """

    input_ids: Integer[TT, " batch seq_len"]
    gaps: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    target_ids: Optional[Integer[TT, " batch seq_len"]]
    max_length: Optional[Integer[TT, " batch"]]
    t: Optional[Float[TT, " batch"]]
    token_weight: Optional[Float[TT, " batch seq_len"]]
    length_weight: Optional[Float[TT, " batch seq_len"]]
    gaps_mask: Optional[Bool[TT, " batch seq_len"]]
    input_positions: Optional[Integer[TT, " batch seq_len"]]


class FlexMDMSeq2SeqPredictionBatch(TypedDict):
    """Input to the MLM for predicting suffix given the prefix."""

    input_ids: Integer[TT, " batch prefix_seq_len"]  # left-padded
    attention_mask: Integer[TT, " batch prefix_seq_len"]
    target_ids: Integer[TT, " batch suffix_seq_len"]


class FlexMDMUncondtionalPredictionBatch:
    """Input to the MLM for unconditional generation.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model. All masks.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]


class FlexMDMLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
    """

    loss: Float[TT, ""]
    unmask_loss: Float[TT, ""]
    insertion_loss: Float[TT, ""]


class FlexMDMModel(Protocol):
    def __call__(
        self,
        input_ids: Integer[TT, " batch seq_len"],
        t: Float[TT, " batch"],
        attention_mask: Optional[Integer[TT, " batch seq_len"]] = None,
        positions: Optional[Integer[TT, " batch seq_len"]] = None,
    ) -> Float[TT, " batch seq_len vocab_size"]: ...


class FlexMDMPredictionDict(TypedDict, total=False):
    """Output of the Predictor for MLM.

    Attributes:
        loss (Optional[Float[TT, "batch"]]): The loss value. Typically None.
        text (List[str]): The batch of generated text with special tokens.
        ids (Integer[TT, " batch seq_len"]): The batch of generated token_ids.
        time_taken (List[float]): Time taken for each prediction.
        history (List[List[Tuple[str, float, int]]]): Generation history for each batch element.
            Each tuple contains (text, confidence, step_number).
        output_start_idx (Integer[TT, " batch"]): The index of the first token in the output.
    """

    loss: Optional[Float[TT, ""]]
    text: List[str]
    ids: Integer[TT, " batch seq_len"]
    time_taken: List[float]
    history: List[List[Tuple[str, float, int]]]
    output_start_idx: int
