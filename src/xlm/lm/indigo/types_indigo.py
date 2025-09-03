from typing import Optional, Protocol, TypedDict
from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


class IndigoBatch(TypedDict):
    """Input batch for training the Indigo model in non seq2seq mode.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): Permuted input ids. The permutation is the order of insertion. This is $z$ in our notation.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for valid tokens, 0 for padding tokens.
        target_ids (Integer[TT, "batch seq_len"]): Target ids. Same as input_ids but shifted by 1, and -100 for pad tokens.
        pi: (Integer[TT, "batch seq_len"]): (alias x_to_z) Indices such that $z = x[x_to_z]$.

    Here is an example:

    input_ids:      BOS  EOS  z1  ...  zn  EOD
    attention_mask: 1    1    1   ...  1   1
    target_ids:     EOS  z1   z2  ...  EOD  -
    pi:             0   n+1   r1  ...  rn   n+2
    lpl:            -   -     x1  ...  xn   1
    rpl:            -   -     y1  ...  yn   0

    """

    input_ids: Integer[TT, " batch seq_len"]  # permuted
    attention_mask: Integer[TT, " batch seq_len"]
    target_ids: Integer[TT, "batch seq_len"]
    pi: Integer[TT, "batch seq_len"]
    # left_pointer_labels: Integer[TT, "batch steps"]
    # right_pointer_labels: Integer[TT, "batch steps"]


class IndigoPredBatch(TypedDict):
    """Input batch for predicting using the Indigo model"""

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    target_ids: Optional[Integer[TT, "batch seq_len"]]


class IndigoSeq2SeqBatch(TypedDict):
    """Input batch for training the Indigo model in seq2seq mode.
    Follows similar format at ARLM but with additional permuation related fields.
    input_ids:  p0  ...  pn-1  BOS z0 ... zm-1 EOD PAD   PAD,      where z0, ..., zm-1 are permuted
    pi:         0   ...  n-1   n   r0 ... rm-1 n+m n+m+1 n+m+2,       where n < r1, ..., rm <= n + m-1 are pi indices
    tgt_ids:    -  ...   -     z0  z1 ... EOD  -    -     -,         where - represents -100 ie. no-loss-tokens

    # These are created later in the model.
    lpl:        -  ...   -     x0  x1 ... -    -    -     -,            where x0, ..., xn-1 are the left pointer labels (intergers in (n, n+m-1])
    rpl:        -  ...   -     y0  y1 ... -    -    -     -,              where y0, ..., ym-1 are the right pointer labels

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): Permuted input ids (prompt + target).
            The permutation is the order of insertion. This is $z$ in our notation. The prompt is not permuted.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for valid tokens, 0 for padding tokens.
        target_ids (Integer[TT, "batch seq_len"]): Same as input_ids but shifted by 1, and -100 for pad tokens.
        pi: (Integer[TT, "batch seq_len"]): (alias x_to_z) Indices such that $z = x[x_to_z]$.
        left_pointer_labels (Integer[TT, "batch steps"]): Pointer labels in generation trajectory
        right_pointer_labels (Integer[TT, "batch steps"]): Pointer labels in generation trajectory
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    target_ids: Integer[TT, "batch seq_len"]
    pi: Integer[TT, "batch seq_len"]
    # left_pointer_labels: Integer[TT, "batch steps"]
    # right_pointer_labels: Integer[TT, "batch steps"]


class IndigoSeq2SeqPredBatch(TypedDict):
    """Input batch for predicting the Indigo model in seq2seq mode."""

    input_ids: Integer[TT, " batch prefix_len"]
    attention_mask: Integer[TT, " batch prefix_len"]
    target_ids: Integer[TT, "batch prefix_len+target_seq_len"]
    pi: Integer[TT, "batch prefix_len"]


class IndigoLossDict(TypedDict):
    """Output of the Indigo loss function."""

    loss: Float[TT, ""]
    token_loss: Float[TT, ""]
    position_loss: Float[TT, ""]


class IndigoPredictionDict(TypedDict):
    """Output of the Indigo predictor."""

    text: list[str]
    ids: Integer[TT, " batch seq_len"]
    relative_matrix: Optional[Integer[TT, "batch pred_plus2 pred_plus2"]]
    # pointer softmax
    pointer_scores: Optional[Float[TT, "batch steps max_slots"]]
    word_scores: Optional[Float[TT, "batch steps vocab"]]
    absolute_positions: Optional[Integer[TT, "batch pred_plus2"]]


class IndigoModelProtocol(Protocol):
    def __call__(
        self,
        x_t: Integer[TT, " batch seq_len"],
        pi: Optional[Integer[TT, " batch seq_len"]] = None,
        attention_mask: Optional[Bool[TT, " batch seq_len seq_len"]] = None,
        rel_matrix: Optional[Integer[TT, " batch seq_len seq_len"]] = None,
    ): ...

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        pi: Optional[Integer[TT, " *batch seq_len"]] = None,
        attention_mask: Optional[Bool[TT, " *batch seq_len seq_len"]] = None,
        rel_matrix: Optional[Integer[TT, " *batch seq_len seq_len"]] = None,
    ): ...

    def get_position_logits(
        self,
        hidden_states: Float[TT, " *batch seq_len d_model"],
        target_ids: Integer[TT, " *batch seq_len"],
    ): ...
