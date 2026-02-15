"""Unit tests for metric wrappers in ``xlm.metrics``."""

import pytest
import torch
from torchmetrics import MeanMetric

from xlm.metrics import MetricWrapper


def _identity_update_fn(batch, loss_dict, tokenizer, **kwargs):
    """A trivial update_fn that passes ``value`` straight through."""
    return {"value": loss_dict["loss"]}


class TestMetricWrapper:
    """Tests for :class:`MetricWrapper`."""

    @pytest.fixture()
    def wrapper(self):
        return MetricWrapper(
            name="test_loss",
            metric=MeanMetric(),
            update_fn=_identity_update_fn,
            prefix="train/",
        )

    def test_update_and_compute(self, wrapper):
        batch = {}
        loss_dict = {"loss": torch.tensor(1.5)}
        wrapper.update(batch, loss_dict)
        value = wrapper.metric.compute()
        assert value.item() == pytest.approx(1.5)
