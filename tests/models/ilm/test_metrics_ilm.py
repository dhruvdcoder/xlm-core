"""Unit tests for :mod:`ilm.metrics_ilm`."""

import torch

from ilm.metrics_ilm import (
    length_loss_metric_update_fn,
    mean_metric_update_fn,
    token_ce_metric_update_fn,
)


class TestMeanMetric:
    def test_returns_batch_loss_mean(self):
        batch_loss = torch.tensor([1.0, 2.0, 3.0])
        out = mean_metric_update_fn({}, {"batch_loss": batch_loss})
        assert torch.allclose(out["value"], torch.tensor(2.0))


class TestLengthLossMetric:
    def test_returns_per_example_length_loss(self):
        v = torch.tensor([0.1, 0.2, 0.3])
        out = length_loss_metric_update_fn(
            {}, {"per_example_length_loss": v}
        )
        assert torch.equal(out["value"], v)


class TestTokenCeMetric:
    def test_returns_per_example_ce(self):
        v = torch.tensor([0.4, 0.5, 0.6])
        out = token_ce_metric_update_fn({}, {"per_example_ce": v})
        assert torch.equal(out["value"], v)
