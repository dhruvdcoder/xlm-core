"""Config for the Dream-Coder FlexMDM (``FlexMDMDreamModel``).

Reference: ``genuine-any-order/FlexMDM/flexmdm/architecture.py`` (Kim et al.,
"From Interface to Inference", arXiv:2607.26504). The backbone config is the
unchanged ``Dream-org/Dream-Coder-v0-Base-7B`` config; the only additions are the
time-embedding hyper-parameters of the AdaLN conditioning.
"""

from xlm.backbones.dream.configuration_base import DreamConfigBase


class FlexMDMDreamConfig(DreamConfigBase):
    """Dream backbone config + FlexMDM AdaLN time-embedding settings.

    ``use_time_embedding`` / ``time_conditioning`` default to the FlexMDM
    reference parametrization (one shared ``(scale, shift)`` modulation per
    decoder layer, driven by a conditioning vector of size ``hidden_size``).
    """

    def __init__(
        self,
        *args,
        use_time_embedding: bool = True,
        time_conditioning: str = "flexmdm_shared2",
        time_scale: float = 1000.0,
        time_max_period: int = 10000,
        **kwargs,
    ):
        super().__init__(
            *args,
            use_time_embedding=use_time_embedding,
            time_conditioning=time_conditioning,
            **kwargs,
        )
        self.time_scale = float(time_scale)
        self.time_max_period = int(time_max_period)
