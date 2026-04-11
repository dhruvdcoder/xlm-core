"""Text utility functions for xlm."""

from typing import List, Optional
from xlm.datamodule import Tokenizer


def remove_trailing_pads(
    text: str,
    tokenizer: Tokenizer,
    tokens_to_remove: Optional[List[str]] = None,
) -> str:
    """Remove trailing special tokens from decoded text.

    For each token, strips either a spaced suffix (``" {token}"``) or the bare
    token, matching the original pad-only behavior. Repeats until no listed
    suffix matches the end of the string.

    Args:
        text: Decoded text string that may contain trailing special tokens
        tokenizer: Tokenizer instance (used for default ``pad_token``)
        tokens_to_remove: Strings to strip from the end; defaults to ``pad_token``

    Returns:
        Text with trailing occurrences of those tokens removed
    """
    tokens_to_remove = tokens_to_remove or [tokenizer.pad_token]
    suffixes: List[str] = []
    for t in tokens_to_remove:
        if not t:
            continue
        suffixes.append(f" {t}")
        suffixes.append(t)
    # Longest first so e.g. " </s>" wins over "</s>" when both could apply
    suffixes = sorted(set(suffixes), key=len, reverse=True)
    if not suffixes:
        return text
    while True:
        for suf in suffixes:
            if text.endswith(suf):
                text = text[: -len(suf)]
                break
        else:
            break
    return text
