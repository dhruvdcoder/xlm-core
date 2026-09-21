"""IWSLT14 German→English (RDM text + joint CharBPE) preprocessing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from xlm.tasks.iwslt14_de_en.mt_eval import Iwslt14MtEval

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Frozen HF CharBPE (sha256 tokenizer.json = 975aedcb168b…). Optional override via
# ``IWSLT14_JOINT_CHAR_BPE_V1`` for local rebuilds; packaged default works after pip install.
TOKENIZER_V1_DIR = Path(__file__).resolve().parent / "tokenizer_v1"

__all__ = [
    "TOKENIZER_V1_DIR",
    "joint_char_bpe_v1_dir",
    "iwslt14_mt_preprocess_fn",
    "Iwslt14MtEval",
]


def joint_char_bpe_v1_dir() -> str:
    """Absolute path to the packaged joint CharBPE v1 directory.

    Prefer ``IWSLT14_JOINT_CHAR_BPE_V1`` when set; otherwise the files shipped under
    ``xlm.tasks.iwslt14_de_en.tokenizer_v1`` (works for editable and wheel installs).
    """
    override = os.environ.get("IWSLT14_JOINT_CHAR_BPE_V1")
    if override:
        return override
    return str(TOKENIZER_V1_DIR)


def iwslt14_mt_preprocess_fn(
    example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, Any]:
    """Encode DE/EN text to raw BPE id lists (no special tokens).

    Collators own BOS/EOS, truncation, padding, and noise.
    """
    source = example.get("source_text") or ""
    target = example.get("target_text") or ""
    example["prompt_token_ids"] = tokenizer.encode(  # type: ignore[call-arg]
        source, add_special_tokens=False
    )
    example["input_token_ids"] = tokenizer.encode(  # type: ignore[call-arg]
        target, add_special_tokens=False
    )
    return example
