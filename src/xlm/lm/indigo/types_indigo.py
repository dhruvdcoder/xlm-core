from typing import Optional, TypedDict
from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


class IndigoBatch(TypedDict):
    """Input batch for training the Indigo model in non seq2seq mode.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): Permuted input ids. The permutation is the order of insertion. This is $z$ in our notation.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for valid tokens, 0 for padding tokens.
        target_ids (Integer[TT, "batch seq_len"]): Target ids. Same as input_ids but shifted by 1, and -100 for pad tokens.
        pi: (Integer[TT, "batch seq_len"]): (alias x_to_z) Indices such that $z = x[x_to_z]$.
        left_pointer_labels (Integer[TT, "batch steps"]): Pointer labels in generation trajectory
        right_pointer_labels (Integer[TT, "batch steps"]): Pointer labels in generation trajectory
        left_pointer_labels_mask (Bool[TT, "batch steps"]): 1 for valid labels, 0 for padding labels
        right_pointer_labels_mask (Bool[TT, "batch steps"]): 1 for valid labels, 0 for padding labels
    """

    input_ids: Integer[TT, " batch seq_len"]  # permuted
    attention_mask: Integer[TT, " batch seq_len"]
    target_ids: Integer[TT, "batch seq_len"]
    pi: Integer[TT, "batch seq_len"]


class IndigoSeq2SeqBatch(TypedDict):
    """Input batch for training the Indigo model in seq2seq mode."""


class IndigoLossDict(TypedDict):
    """Output of the Indigo loss function."""

    loss: Float[TT, ""]
    batch_loss: Float[TT, "batch"]
    word_loss: Float[TT, ""]
    position_loss: Float[TT, ""]
    word_acc: Float[TT, ""]
    pointer_acc: Float[TT, ""]
    ppl: Float[TT, ""]


class IndigoPredictionDict(TypedDict):
    """Output of the Indigo predictor."""

    text: list[str]
    ids: Integer[TT, " batch seq_len"]
    relative_matrix: Optional[Integer[TT, "batch pred_plus2 pred_plus2"]]
    # pointer softmax
    pointer_scores: Optional[Float[TT, "batch steps max_slots"]]
    word_scores: Optional[Float[TT, "batch steps vocab"]]
    absolute_positions: Optional[Integer[TT, "batch pred_plus2"]]
