"""Unit tests for :mod:`arlm.metrics_arlm`."""

import torch

from arlm.metrics_arlm import (
    mean_metric_update_fn,
    perplexity_metric_update_fn,
    seq2seq_exact_match_update_fn,
    seq2seq_token_accuracy_update_fn,
    sequence_length_metric_update_fn,
    token_nll_metric_update_fn,
    valid_tokens_metric_update_fn,
)


class TestSeq2SeqExactMatch:
    def test_slices_from_output_start_idx(self):
        # ids: shape (2, 6) with prefix length 4 -> 2 generated tokens.
        ids = torch.arange(12).reshape(2, 6)
        loss_dict = {"ids": ids, "output_start_idx": 4}
        batch = {"target_ids": torch.tensor([[10, 11], [16, 17]])}
        out = seq2seq_exact_match_update_fn(batch, loss_dict)
        assert torch.equal(out["pred"], ids[:, 4:])
        assert torch.equal(out["target"], batch["target_ids"])
        assert out["pred_length"] is None
        assert out["target_length"] is None


class TestSeq2SeqTokenAccuracy:
    def test_returns_pred_target_and_full_mask(self):
        ids = torch.arange(12).reshape(2, 6)
        loss_dict = {"ids": ids, "output_start_idx": 4}
        batch = {"target_ids": torch.tensor([[10, 11], [16, 17]])}
        out = seq2seq_token_accuracy_update_fn(batch, loss_dict)
        assert out["pred"].shape == (2, 2)
        assert torch.equal(out["target"], batch["target_ids"])
        assert out["pred_mask"].dtype == torch.bool
        assert out["pred_mask"].all()


class TestMeanMetric:
    def test_returns_loss_value(self):
        loss = torch.tensor(0.5)
        out = mean_metric_update_fn({}, {"loss": loss})
        assert torch.equal(out["value"], loss)


class TestPerplexity:
    def test_perplexity_is_exp_mean_nlls(self):
        nlls = torch.tensor([1.0, 2.0, 3.0])
        out = perplexity_metric_update_fn({}, {"nlls": nlls})
        assert torch.allclose(out["value"], torch.exp(nlls.mean()))


class TestTokenNll:
    def test_returns_nlls(self):
        nlls = torch.randn(4)
        out = token_nll_metric_update_fn({}, {"nlls": nlls})
        assert torch.equal(out["value"], nlls)


class TestSequenceLength:
    def test_counts_attention_mask_per_row(self):
        attn = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        out = sequence_length_metric_update_fn(
            {"attention_mask": attn}, {}
        )
        assert torch.allclose(
            out["value"], torch.tensor([3.0, 2.0, 5.0])
        )


class TestValidTokens:
    def test_excludes_padding_and_ignore_index(self):
        # attention_mask is a bool here so "&" with target_ids != -100 works.
        attn = torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ],
            dtype=torch.bool,
        )
        target_ids = torch.tensor(
            [
                [5, -100, 7, 0],
                [3, 4, 0, 0],
            ]
        )
        out = valid_tokens_metric_update_fn(
            {"attention_mask": attn, "target_ids": target_ids}, {}
        )
        # Row 0: positions [0, 2] are real & not -100 -> 2
        # Row 1: positions [0, 1] are real & not -100 -> 2
        assert torch.allclose(out["value"], torch.tensor([2.0, 2.0]))
