from typing import Optional, TypedDict
from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT


class IndigoBatch(TypedDict):
    """Input batch for the Indigo model."""

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]

    target_ids: Integer[TT, "batch tgt_seq"]        
    target_attention_mask: Integer[TT, "batch tgt_seq"]

    # Generation order, each row is a permutation of [0..tgt_len-1]
    order_indices: Integer[TT, "batch tgt_seq"]     

    # Word (token) labels in generation trajectory 
    word_labels: Integer[TT, "batch steps"]           
    word_labels_mask: Bool[TT, "batch steps"]

    pointer_labels: Integer[TT, "batch steps"]      
    pointer_labels_mask: Bool[TT, "batch steps"]

    # with -1s, 0s, and 1s
    relative_matrix: Integer[TT, "batch final_plus2 final_plus2"]

    # sbsolute positions derived from relative matrix after mapping back 
    absolute_positions: Optional[Integer[TT, "batch final_plus2"]]

    # Sequence length tensors
    target_lengths: Integer[TT, "batch"]             
    trajectory_lengths: Integer[TT, "batch"]         

    # Misc meta, e.g. ["L2R", "R2L", ...] for each example
    order_name: List[str]                            


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
    #pointer softmax
    pointer_scores: Optional[Float[TT, "batch steps max_slots"]]   
    word_scores: Optional[Float[TT, "batch steps vocab"]]
    absolute_positions: Optional[Integer[TT, "batch pred_plus2"]]
