"""Unit tests for :mod:`mdlm.metrics_mdlm`."""

import torch

from mdlm.metrics_mdlm import (
    mean_metric_update_fn,
    seq2seq_exact_match_update_fn,
    seq2seq_token_accuracy_update_fn,
)


class TestSeq2SeqExactMatch:
    def test_slices_and_records_lengths(self):
        ids = torch.arange(12).reshape(2, 6)
        target = torch.tensor([[10, 11], [16, 17]])
        out = seq2seq_exact_match_update_fn(
            {"target_ids": target},
            {"ids": ids, "output_start_idx": 4},
        )
        assert torch.equal(out["pred"], ids[:, 4:])
        assert torch.equal(out["target"], target)
        assert out["pred_length"] == 2
        assert out["target_length"] == 2


class TestSeq2SeqTokenAccuracy:
    def test_full_pred_mask(self):
        ids = torch.arange(12).reshape(2, 6)
        target = torch.tensor([[10, 11], [16, 17]])
        out = seq2seq_token_accuracy_update_fn(
            {"target_ids": target},
            {"ids": ids, "output_start_idx": 4},
        )
        assert out["pred_mask"].shape == (2, 2)
        assert out["pred_mask"].all()


class TestMeanMetric:
    def test_returns_loss(self):
        loss = torch.tensor(0.7)
        out = mean_metric_update_fn({}, {"loss": loss})
        assert torch.equal(out["value"], loss)
