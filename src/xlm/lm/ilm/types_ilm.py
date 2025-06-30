from typing import Optional, Protocol, Tuple, TypedDict, List, Union

from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


from xlm.datamodule import BaseBatch
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class ILMBatch(BaseBatch):
    """Input to the ILM.

    Attributes:
        input_ids (Integer[TT, " batch post_seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch post_seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch post_seq_len"]): 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
        n_drops (Bool[TT, " batch post_seq_len"]): 1 for tokens that are dropped.
        target_ids (Integer[TT, " batch post_seq_len vocab_size"]): The target ids to the model.
        constraint (Optional[Bool[TT, " batch post_seq_len"]]): 1 for tokens that should not be predicted. Mostly used during prediction only.
        cls_position (Optional[Integer[TT, " batch"]]): The position of the CLS token.
    """

    input_ids: Integer[
        TT, " batch post_seq_len"
    ]  # post_seq_len = seq_len after removing the dropped tokens
    attention_mask: Integer[TT, " batch post_seq_len"]
    token_type_ids: Integer[
        TT, " batch post_seq_len"
    ]  # TODO (remove type_ids)
    n_drops: Optional[
        Bool[TT, " batch post_seq_len"]
    ]  # will be the same as target_ids.sum(dim=-1) but is more readily available
    target_ids: Union[
        Integer[TT, " batch post_seq_len vocab_size"],
        Integer[TT, " batch target_seq_len"],
    ]
    target_attention_mask: Optional[Integer[TT, " batch target_seq_len"]]
    cls_position: Optional[
        Integer[TT, " batch"]
    ]  #  We will assume that the CLS token is at the beginning of the sequence if not provided. Will need it for seq2seq setting.
    # TODO: This has to be generalised to a cls mask to support per-gap CLS tokens.
    constraint: Optional[Bool[TT, " batch post_seq_len"]]


class ILMSeq2SeqPredictionBatch:
    """Input to the ILM for predicting suffix given the prefix.
    Note that the target_ids are different from the ILMBatch

    Attributes:
        input_ids (Integer[TT, " batch prefix_seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch prefix_seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch prefix_seq_len"]): 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
        target_ids (Integer[TT, " batch suffix_seq_len"]): The target ids to the model.
    """

    input_ids: Integer[TT, " batch prefix_seq_len"]
    attention_mask: Integer[TT, " batch prefix_seq_len"]
    token_type_ids: Integer[TT, " batch prefix_seq_len"]
    target_ids: Integer[TT, " batch suffix_seq_len"]


class ILMUncondtionalPredictionBatch:
    """Input to the ILM for unconditional generation.

    Attributes:
        input_ids (Integer[TT, " batch prefix_seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch prefix_seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch prefix_seq_len"]): 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
    """

    input_ids: Integer[TT, " batch 2"]
    attention_mask: Integer[TT, " batch 2"]
    token_type_ids: Integer[TT, " batch 2"]


class ILMLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
        batch_loss (Float[TT, " batch"]): Loss value for each example in the batch.
    """

    loss: Float[TT, ""]
    batch_loss: Float[TT, " batch"]
    per_example_length_loss: Optional[Float[TT, " batch"]]
    per_example_ce: Float[TT, " batch"]
    length_logits: Optional[Float[TT, " batch max_length"]]
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
    ) -> Tuple[
        TokenLogitsType,
        Optional[Union[LengthLogitsType, ClassificationLogitsType]],
    ]: ...

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
