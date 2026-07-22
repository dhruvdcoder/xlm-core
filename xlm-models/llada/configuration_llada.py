"""LLaDA variant config (reference: GSAI-ML/LLaDA-8B-{Base,Instruct} Hub)."""

from xlm.backbones.llada.configuration_llada import LLaDAConfigBase


class LLaDAConfig(LLaDAConfigBase):
    """Configuration for LLaDA checkpoints.

    Base and Instruct share the same architecture; the experiment's
    ``hub.repo_id`` selects which initial weights are loaded.

    NOTE: the explicit ``__init__`` is required with transformers >= 5, which
    otherwise replaces the inherited ``__init__`` of config subclasses with a
    generated dataclass one, silently skipping ``LLaDAConfigBase``'s
    ModelConfig-default handling.
    """

    def __init__(self, use_cache: bool = False, **kwargs):
        super().__init__(use_cache=use_cache, **kwargs)
