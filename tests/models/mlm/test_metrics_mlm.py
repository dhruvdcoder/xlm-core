"""Unit tests for :mod:`mlm.metrics_mlm`."""

import torch

from mlm.metrics_mlm import (
    exact_match_update_fn,
    infill_token_accuracy_update_fn,
    mean_metric_update_fn,
    seq2seq_exact_match_update_fn,
    seq2seq_token_accuracy_update_fn,
)


class TestExactMatch:
    def test_returns_pred_and_target_with_none_lengths(self):
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        target = torch.tensor([[1, 2, 3], [4, 5, 0]])
        out = exact_match_update_fn(
            {"target_ids": target}, {"ids": ids}
        )
        assert torch.equal(out["pred"], ids)
        assert torch.equal(out["target"], target)
        assert out["pred_length"] is None
        assert out["target_length"] is None


class TestInfillTokenAccuracy:
    def test_pred_mask_is_input_equals_mask_token(self, simple_tokenizer):
        mask_id = simple_tokenizer.mask_token_id
        input_ids = torch.tensor(
            [
                [10, mask_id, 11, mask_id],
                [mask_id, 12, 13, 14],
            ]
        )
        ids = torch.tensor([[10, 99, 11, 100], [88, 12, 13, 14]])
        target = torch.tensor([[10, 50, 11, 60], [70, 12, 13, 14]])
        out = infill_token_accuracy_update_fn(
            {"input_ids": input_ids, "target_ids": target},
            {"ids": ids},
            tokenizer=simple_tokenizer,
        )
        # pred_mask must mirror input == mask token positions.
        expected_mask = input_ids == mask_id
        assert torch.equal(out["pred_mask"], expected_mask)
        assert torch.equal(out["pred"], ids)
        assert torch.equal(out["target"], target)


class TestSeq2SeqExactMatch:
    def test_slices_from_output_start_idx(self):
        ids = torch.arange(12).reshape(2, 6)
        loss_dict = {"ids": ids, "output_start_idx": 4}
        target = torch.tensor([[10, 11], [16, 17]])
        out = seq2seq_exact_match_update_fn(
            {"target_ids": target}, loss_dict
        )
        assert torch.equal(out["pred"], ids[:, 4:])
        assert torch.equal(out["target"], target)
        assert out["pred_length"] == 2
        assert out["target_length"] == 2


class TestSeq2SeqTokenAccuracy:
    def test_pred_mask_is_all_true(self):
        ids = torch.arange(12).reshape(2, 6)
        loss_dict = {"ids": ids, "output_start_idx": 4}
        target = torch.tensor([[10, 11], [16, 17]])
        out = seq2seq_token_accuracy_update_fn(
            {"target_ids": target}, loss_dict
        )
        assert out["pred_mask"].dtype == torch.bool
        assert out["pred_mask"].all()


class TestMeanMetric:
    def test_returns_loss_value(self):
        loss = torch.tensor(0.25)
        out = mean_metric_update_fn({}, {"loss": loss})
        assert torch.equal(out["value"], loss)
