"""Unit tests for the ILM predictor."""

import pytest

from ilm.predictor_ilm import ILMPredictor


class TestILMPredictor:
    """Tests for :class:`ILMPredictor`.

    Full prediction requires a wired model + noise schedule + collated batch.
    Construction / validation tests go here; integration tests marked slow.
    """

    @pytest.mark.slow
    def test_predict_returns_expected_keys(self):
        """Placeholder: requires a fully wired ILM predictor + batch."""
        pytest.skip("Requires full ILM wiring -- implement when ready")

    @pytest.mark.slow
    def test_predict_ids_in_vocab_range(self):
        pytest.skip("Requires full ILM wiring -- implement when ready")
