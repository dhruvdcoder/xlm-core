"""Unit tests for the ILM loss function."""

import pytest

from ilm.loss_ilm import ILMLossWithMaskedCE


class TestILMLossWithMaskedCE:
    """Tests for :class:`ILMLossWithMaskedCE`.

    The ILM loss requires a fully collated :class:`ILMBatch` with sparse
    ``target_ids``, ``n_drops``, and ``cls_position``.  Building these from
    scratch is non-trivial, so the tests below verify construction and
    configuration guards; full integration tests should be added once a tiny
    ILM collation fixture exists.
    """

    def test_stopping_class_weight_requires_binary_ce(self):
        with pytest.raises(ValueError):
            ILMLossWithMaskedCE(
                length_loss="ce",
                stopping_class_weight=0.5,
            )

    def test_loss_on_padding_raises(self):
        with pytest.raises(ValueError, match="loss_on_padding"):
            ILMLossWithMaskedCE(loss_on_padding=True, length_loss="ce")

    @pytest.mark.slow
    def test_loss_fn_with_full_batch(self):
        """Placeholder: requires a realistic ILMBatch fixture."""
        pytest.skip("Requires full ILM collation pipeline -- implement when ready")
