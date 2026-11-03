# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Shared Dream-family backbone configuration (subclass per variant in xlm-models)."""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DreamConfigBase(PretrainedConfig):
    """Architecture fields shared by Dream and DreamOn decoder LMs."""

    model_type = "Dream"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        mask_token_id=151666,
        pad_token_id=151643,
        use_time_embedding=False,
        d_cond=None,
        time_conditioning="rmsnorm4",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.use_time_embedding = use_time_embedding
        # How the time conditioning vector ``c`` modulates the decoder layers:
        # - "rmsnorm4": ``DreamBaseModel.sigma_map`` (TimestepEmbedder) builds
        #   ``c`` of size ``d_cond`` (default hidden_size // 2) and every layer has
        #   a 4-chunk modulation (separate shift/scale for attention and MLP norms).
        # - "flexmdm_shared2": ``c`` of size ``d_cond`` (default hidden_size) is
        #   computed outside the backbone (e.g. ``flexmdm.FlexMDMDreamModel``) and
        #   passed to ``forward(c=...)``; every layer has one 2-chunk
        #   ``(scale, shift)`` modulation shared by both norms, matching the
        #   FlexMDM (Dream-Coder) reference ``temb_mods``.
        if time_conditioning not in ("rmsnorm4", "flexmdm_shared2"):
            raise ValueError(
                f"Unknown time_conditioning={time_conditioning!r}; expected "
                "'rmsnorm4' or 'flexmdm_shared2'."
            )
        self.time_conditioning = time_conditioning
        if d_cond is not None:
            self.d_cond = d_cond
        elif time_conditioning == "flexmdm_shared2":
            self.d_cond = hidden_size
        else:
            self.d_cond = hidden_size // 2
