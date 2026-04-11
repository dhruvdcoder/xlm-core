"""DreamOn variant config (expand token and other DreamOn Hub fields)."""

from xlm.backbones.dream.configuration_base import DreamConfigBase


class DreamOnConfig(DreamConfigBase):
    def __init__(self, **kwargs):
        expand_token_id = kwargs.pop("expand_token_id", 151667)
        super().__init__(**kwargs)
        self.expand_token_id = expand_token_id
