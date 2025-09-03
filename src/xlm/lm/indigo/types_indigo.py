from typing import Optional, TypedDict
from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


class IndigoBatch(TypedDict):
    """Input batch for the Indigo model."""

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    # TODO (URV): Add other fields as needed for indigo model


class IndigoLossDict(TypedDict):
    """Output of the Indigo loss function."""

    loss: Float[TT, ""]
    batch_loss: Float[TT, " batch"]
    # TODO (URV): Add other metrics as needed


class IndigoPredictionDict(TypedDict):
    """Output of the Indigo predictor."""

    text: list[str]
    ids: Integer[TT, " batch seq_len"]
    # TODO (URV): Add other fields as needed
