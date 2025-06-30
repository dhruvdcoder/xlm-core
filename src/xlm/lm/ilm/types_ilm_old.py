from typing import Optional, Protocol, Tuple, TypedDict, List

from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


from xlm.datamodule import BaseBatch
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class ILMBatch(BaseBatch):
    """Input to the ILM.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
        drop (Bool[TT, " batch seq_len"]): 1 for tokens that are dropped.
        target_ids (Integer[TT, " batch seq_len vocab_size"]): The target ids to the model.
        constraint (Optional[Bool[TT, " batch seq_len"]]): 1 for tokens that should not be predicted. Mostly used during prediction only.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    token_type_ids: Integer[TT, " batch seq_len"]
    drop: Bool[TT, " batch seq_len"]
    target_ids: Integer[TT, " batch seq_len vocab_size"]
    constraint: Optional[Bool[TT, " batch seq_len"]]


class NewILMBatch(BaseBatch):
    """Input to the ILM.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
        drop (Bool[TT, " batch seq_len"]): 1 for tokens that are dropped.
        target_ids (Integer[TT, " batch seq_len vocab_size"]): The target ids to the model.
        constraint (Optional[Bool[TT, " batch seq_len"]]): 1 for tokens that should not be predicted. Mostly used during prediction only.
    """

    input_ids: Integer[TT, " batch post_seq_len"] # post_seq_len = seq_len after removing the dropped tokens
    attention_mask: Integer[TT, " batch post_seq_len"]
    token_type_ids: Integer[TT, " batch post_seq_len"]
    n_drops: Bool[TT, " batch post_seq_len"] # will be the same as target_ids.sum(dim=-1)
    target_ids: Integer[TT, " batch post_seq_len vocab_size"]
    constraint: Optional[Bool[TT, " batch post_seq_len"]]


class ILMLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
        batch_loss (Float[TT, " batch"]): Loss value for each example in the batch.
    """

    loss: Float[TT, ""]
    batch_loss: Float[TT, " batch"]


class ILMWithLengthClassificationLossDict(ILMLossDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
        per_example_length_loss (Float[TT, " batch"]): Length loss value for each example in the batch.
        per_example_ce (Float[TT, " batch"]): Token prediction loss value for each example in the batch.
        batch_loss (Float[TT, ""]): Combined loss value (length and token prediction) for each example in the batch.
        length_logits (Float[TT, " batch max_length"]): The length logits for each example in the batch.
        n_drops (Integer[TT, " batch"]): The number of tokens dropped for each example in the batch.
    """

    loss: Float[TT, ""]
    per_example_length_loss: Float[TT, " batch"]
    per_example_ce: Float[TT, " batch"]
    batch_loss: Float[TT, ""]
    length_logits: Float[TT, " batch max_length"]
    n_drops: Integer[TT, " batch"]


class ILMWithStoppingClassificationLossDict(ILMLossDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
        per_example_stopping_loss (Float[TT, " batch"]): Stopping loss value for each example in the batch.
        per_example_ce (Float[TT, " batch"]): Token prediction loss value for each example in the batch.
        batch_loss (Float[TT, ""]): Combined loss value (stopping and token prediction) for each example in the batch.
        stopping_logits (Float[TT, " batch 2"]): The stopping logits for each example in the batch.
        n_drops (Integer[TT, " batch"]): The number of tokens dropped for each example in the batch.
    """

    loss: Float[TT, ""]
    per_example_stopping_loss: Float[TT, " batch"]
    per_example_ce: Float[TT, " batch"]
    batch_loss: Float[TT, ""]
    stopping_logits: Float[TT, " batch 2"]
    n_drops: Integer[TT, " batch"]


TokenLogitsType = Float[TT, " batch seq_len vocab_size"]
LengthLogitsType = Float[TT, " batch max_length"]
ClassificationLogitsType = Float[TT, " batch num_classes"]


class ILMModel(Protocol):
    def __call__(
        self,
        x_t: Integer[TT, " batch seq_len"],
        non_drop_non_pad: Integer[TT, " batch seq_len"],
        positions: Integer[TT, " batch seq_len"],
        token_type_ids: Optional[Integer[TT, " batch seq_len"]] = None,
    ) -> Tuple[TokenLogitsType, None]: ...


class ILMModelWithStoppingClassification(Protocol):
    def __call__(
        self,
        x_t: Integer[TT, " batch seq_len"],
        non_drop_non_pad: Integer[TT, " batch seq_len"],
        positions: Integer[TT, " batch seq_len"],
        token_type_ids: Optional[Integer[TT, " batch seq_len"]] = None,
    ) -> Tuple[TokenLogitsType, ClassificationLogitsType]: ...


class ILMModelWithLengthClassification(Protocol):
    def __call__(
        self,
        x_t: Integer[TT, " batch seq_len"],
        non_drop_non_pad: Integer[TT, " batch seq_len"],
        positions: Integer[TT, " batch seq_len"],
        token_type_ids: Optional[Integer[TT, " batch seq_len"]] = None,
    ) -> Tuple[TokenLogitsType, LengthLogitsType]: ...

    def get_mean_delta_l(
        self, logits: ClassificationLogitsType
    ) -> Float[TT, " batch"]: ...


class ILMPredictionDict(TypedDict):
    """Output of the Predictor for ILM.

    Attributes:
        loss (Optional[Float[TT, "batch"]]): The loss value. Typically None.
        text (List[str]): The batch of generated text without special tokens.
        text_with_spl_tokens (List[str]): The batch of generated text with special tokens.
        ids (Integer[TT, " batch seq_len"]): The batch of generated token_ids.
        attention_mask (Bool[TT, " batch seq_len"]): Attention mask accompanying the generated ids.
        positions (Integer[TT, " batch seq_len"]): The batch of positions of the generated tokens accompanying the ids.
        history (List[List[Tuple[str, float, int]]]): The batch of history.
            Each entry is a list of tuples, where each tuple contains
            (current_string, time, step_number) of when some change is made to the generated string.
    """

    loss: Optional[Float[TT, ""]]
    text: List[str]
    text_with_spl_tokens: List[str]
    ids: Integer[TT, " batch seq_len"]
    attention_mask: Bool[TT, " batch seq_len"]
    positions: Integer[TT, " batch seq_len"]
    history: List[List[Tuple[str, float, int]]]
    time_taken: List[float]
