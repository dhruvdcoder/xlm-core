"""DreamOn model: backbone + variable-canvas diffusion generation."""


from transformers import PretrainedConfig
from xlm.backbones.dream.modeling_dream import DreamModelCore
from .configuration_dreamon import DreamOnConfig


class DreamOnModel(DreamModelCore):
    """DreamOn decoder with expand/delete `diffusion_generate`."""

    config_class = DreamOnConfig

    def get_named_params_for_weight_decay(self):
        # all parameters except biases and layer-norm parameters
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        # biases and layer-norm parameters
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                yield (name, param)
