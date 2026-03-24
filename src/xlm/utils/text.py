"""Text utility functions for xlm."""

from xlm.datamodule import Tokenizer


def remove_trailing_pads(text: str, tokenizer: Tokenizer) -> str:
    """Remove trailing pad tokens from decoded text.
    
    Args:
        text: Decoded text string that may contain trailing pad tokens
        tokenizer: Tokenizer instance containing pad_token
        
    Returns:
        Text with trailing pad tokens removed
    """
    suf = f" {tokenizer.pad_token}"
    if not text.endswith(suf):
        suf = f"{tokenizer.pad_token}"
        if not text.endswith(suf):
            return text
    lsuf = len(suf)
    while text.endswith(suf):
        text = text[:-lsuf]
    return text
