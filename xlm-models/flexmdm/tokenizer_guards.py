"""Tokenizer preconditions for FlexMDM training and prediction."""

from xlm.datamodule import Tokenizer


def require_distinct_pad_and_eos(tokenizer: Tokenizer) -> None:
    """Raise if pad and eos share an id (breaks FlexMDM length / insertion logic)."""
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None or eos_id is None:
        raise ValueError("FlexMDM requires pad_token_id and eos_token_id to be set.")
    if pad_id == eos_id:
        raise ValueError(
            f"FlexMDM requires pad_token_id != eos_token_id (both are {pad_id}). "
            "Add pad_token: '<|pad|>' under global_components.tokenizer.special_tokens."
        )
