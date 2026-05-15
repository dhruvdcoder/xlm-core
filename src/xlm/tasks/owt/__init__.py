from typing import Dict, Any
from transformers import PreTrainedTokenizerBase


def preprocess_fn(
    example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, Any]:
    example["token_ids"] = tokenizer.encode(  # type: ignore
        example["text"], add_special_tokens=False
    )
    return example
