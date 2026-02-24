"""Unit tests for ILM collators."""

import pytest

from ilm.datamodule_ilm import DefaultILMCollator


class TestDefaultILMCollator:
    """Tests for :class:`DefaultILMCollator`.

    The ILM collator performs token-drop masking and builds sparse target
    tensors, which requires a full noise schedule.  Placeholder tests are
    provided; fill in once a real noise schedule fixture is available.
    """

    @pytest.mark.slow
    def test_output_has_expected_keys(self):
        pytest.skip(
            "DefaultILMCollator needs a real NoiseSchedule -- implement when ready"
        )

    @pytest.mark.slow
    def test_output_shapes(self):
        pytest.skip(
            "DefaultILMCollator needs a real NoiseSchedule -- implement when ready"
        )
