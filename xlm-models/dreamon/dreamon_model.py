"""DreamOn model: backbone + variable-canvas diffusion generation."""


from transformers import PretrainedConfig
from xlm.backbones.dream.modeling_dream import DreamModelCore
from .configuration_dreamon import DreamOnConfig


class DreamOnModel(DreamModelCore):
    """DreamOn decoder with expand/delete `diffusion_generate`."""

    config_class = DreamOnConfig
