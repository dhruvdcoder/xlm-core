"""Unit tests for MDLM collators."""

import pytest
import torch

from mdlm.datamodule_mdlm import DefaultMDLMCollator


class TestDefaultMDLMCollator:
    """Tests for :class:`DefaultMDLMCollator`.

    Note: DefaultMDLMCollator requires a *real* noise schedule (not DummyNoiseSchedule)
    because it calls ``noise_schedule.sample_t()`` and ``noise_schedule.forward()``
    during collation. Tests that need a real schedule should provide one or skip.
    """

    @pytest.mark.slow
    def test_output_has_noise_fields(self):
        """Placeholder: requires a real noise schedule to collate."""
        pytest.skip(
            "DefaultMDLMCollator needs a real NoiseSchedule -- implement when ready"
        )

    @pytest.mark.slow
    def test_output_shapes(self):
        pytest.skip(
            "DefaultMDLMCollator needs a real NoiseSchedule -- implement when ready"
        )
