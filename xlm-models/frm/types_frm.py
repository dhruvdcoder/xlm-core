from typing import List, Optional, Protocol, TypedDict

from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT
from typing_extensions import NotRequired


class FRMBatch(TypedDict, total=False):
    """Training / infill batch for FRM.

    ``input_ids`` is the clean sequence ``y`` at train time (solution) and the
    prompt (with mask blanks) at infill prediction time.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    target_ids: Optional[Integer[TT, " batch seq_len"]]
    clamp_mask: NotRequired[Bool[TT, " batch seq_len"]]
    clue_ids: NotRequired[Integer[TT, " batch seq_len"]]


class FRMSeq2SeqPredictionBatch(TypedDict, total=False):
    """Prefix-only batch for seq2seq generation."""

    input_ids: Integer[TT, " batch prefix_seq_len"]
    attention_mask: Integer[TT, " batch prefix_seq_len"]
    target_ids: NotRequired[Integer[TT, " batch suffix_seq_len"]]
    clamp_mask: NotRequired[Bool[TT, " batch prefix_seq_len"]]
    clue_ids: NotRequired[Integer[TT, " batch prefix_seq_len"]]


class FRMLossDict(TypedDict):
    loss: Float[TT, ""]


class FRMModel(Protocol):
    num_embeddings: int

    def __call__(
        self,
        x_t: Float[TT, " batch seq_len vocab_size"],
        t: Float[TT, " batch"],
        attention_mask: Optional[Bool[TT, " batch seq_len"]] = None,
        positions: Optional[Integer[TT, " batch seq_len"]] = None,
        s: Optional[Float[TT, " batch seq_len vocab_size"]] = None,
        clue_ids: Optional[Integer[TT, " batch seq_len"]] = None,
        clamp_mask: Optional[Bool[TT, " batch seq_len"]] = None,
    ) -> Float[TT, " batch seq_len vocab_size"]: ...


class FRMPredictionDict(TypedDict):
    loss: Optional[Float[TT, ""]]
    text: List[str]
    ids: Integer[TT, " batch seq_len"]
    time_taken: List[float]
    output_start_idx: int
    steps_taken: Integer[TT, " batch"]
