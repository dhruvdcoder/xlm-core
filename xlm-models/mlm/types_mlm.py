from typing import Optional, Protocol, Tuple, TypedDict, List, Union

from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class MLMBatch(TypedDict, total=False):
    """Input to the MLM.

    Attributes:
        input_ids: The (possibly masked) input token ids.
        attention_mask: Boolean mask — shape ``(batch, seq_len)`` for standard
            padded batches (True = valid token), or ``(batch, seq_len, seq_len)``
            for packed sequences with per-protein block attention.
        target_ids: Ground-truth token ids (masks replaced with original tokens).
        positions: Optional per-token position indices.  When present (packed
            sequences) they reset to 0 at the start of each protein; when absent
            ``MLMLoss`` derives positions from the 1-D attention mask.
        segment_ids: Optional integer tensor ``(batch, seq_len)`` where each
            position holds the 0-based index of the protein it belongs to within
            the packed block.  Produced by ``PackedMLMCollator`` and consumed by
            ``RotaryTransformerMLMModel`` to build a ``BlockMask`` for
            FlexAttention-based document masking.
        fixed_positions_mask: Optional boolean mask marking positions that should
            not be masked (used by infilling collators).
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: TT  # 2-D (batch, seq_len) or 3-D (batch, seq_len, seq_len)
    target_ids: Optional[Integer[TT, " batch seq_len"]]
    positions: Optional[Integer[TT, " batch seq_len"]]
    segment_ids: Optional[Integer[TT, " batch seq_len"]]
    fixed_positions_mask: Optional[Bool[TT, " batch seq_len"]]


class MLMSeq2SeqPredictionBatch(TypedDict):
    """Input to the MLM for predicting suffix given the prefix."""

    input_ids: Integer[TT, " batch prefix_seq_len"]  # left-padded
    attention_mask: Integer[TT, " batch prefix_seq_len"]
    target_ids: Integer[TT, " batch suffix_seq_len"]


class MLMUncondtionalPredictionBatch:
    """Input to the MLM for unconditional generation.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model. All masks.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]


class MLMLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
    """

    loss: Float[TT, ""]


class MLMModel(Protocol):
    def __call__(
        self,
        input_ids: Integer[TT, " batch seq_len"],
        attention_mask: Integer[TT, " batch seq_len"],
        target_ids: Optional[Integer[TT, " batch seq_len"]] = None,
    ) -> Float[TT, " batch seq_len vocab_size"]: ...


class MLMPredictionDict(TypedDict):
    """Output of the Predictor for MLM.

    Attributes:
        loss (Optional[Float[TT, "batch"]]): The loss value. Typically None.
        text (List[str]): The batch of generated text with special tokens.
        ids (Integer[TT, " batch seq_len"]): The batch of generated token_ids.
        time_taken (List[float]): Time taken for each prediction.
        output_start_idx (Integer[TT, " batch"]): The index of the first token in the output.
        steps_taken (List[int]): Number of steps taken per sample.
    """

    loss: Optional[Float[TT, ""]]
    text: List[str]
    ids: Integer[TT, " batch seq_len"]
    time_taken: List[float]
    output_start_idx: int
    steps_taken: List[int]
