"""LLaDA-family backbone (modeling + LLaDAConfigBase).

Ported from the ``GSAI-ML/LLaDA-8B-Base`` Hub repo with state-dict keys kept
identical to the checkpoint (strict load, no remapping). The tokenizer is a
stock ``PreTrainedTokenizerFast``; load it via
``xlm.datamodule.load_auto_tokenizer`` (no custom tokenizer class needed).
"""

from xlm.backbones.llada.configuration_llada import LLaDAConfigBase, ModelConfig
from xlm.backbones.llada.modeling_llada import (
    LLaDABlock,
    LLaDALlamaBlock,
    LLaDAModel,
    LLaDAModelLM,
    LLaDASequentialBlock,
)

__all__ = [
    "LLaDAConfigBase",
    "ModelConfig",
    "LLaDABlock",
    "LLaDALlamaBlock",
    "LLaDASequentialBlock",
    "LLaDAModel",
    "LLaDAModelLM",
]
