"""Unit tests for the MDLM predictor."""

import pytest
import torch

from mdlm.predictor_mdlm import MDLMPredictor


class TestMDLMPredictor:
    """Tests for :class:`MDLMPredictor`."""

    @pytest.mark.slow
    def test_predict_returns_expected_keys(self):
        """Requires a real NoiseSchedule (DummyNoiseSchedule raises on call)."""
        pytest.skip(
            "MDLMPredictor.predict() needs a real NoiseSchedule -- "
            "implement when a test-friendly schedule is available"
        )
