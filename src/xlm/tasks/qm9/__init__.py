from typing import Dict, Any
from transformers import AutoTokenizer


def preprocess_fn(
    example: Dict[str, Any], tokenizer
) -> Dict[str, Any]:
    text = example['canonical_smiles']
    example["token_ids"] = tokenizer.encode(  # type: ignore
        text, add_special_tokens=False
    )
    return example

def get_tokenizer(pretrained_model_name_or_path: str ):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,trust_remote_code=True)
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(
                'Tokenizer must have a bos_token or '
                f'cls_token: {tokenizer}')
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(
                'Tokenizer must have a eos_token '
                f'or sep_token: {tokenizer}')
        tokenizer.eos_token = tokenizer.sep_token
    tokenizer.full_vocab_size = tokenizer.__len__()
    return tokenizer