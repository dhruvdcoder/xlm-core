"""Unit tests for :mod:`ilm.nn`."""

import torch

from ilm.nn import (
    _remove_tokens,
    general_sample_over_last_two_dims,
    log_softmax_last_two_dims,
    masked_ce_last_two_dims,
    max_over_last_two_dims,
    remove_tokens,
    sample_over_last_two_dims,
    topk_over_last_two_dims,
)


PAD = 0
MASK = 99


class TestRemoveTokens:
    def test_int_mask_token(self):
        ids = torch.tensor(
            [
                [1, MASK, 2, MASK, 3],
                [MASK, 4, MASK, 5, MASK],
            ]
        )
        out = remove_tokens(ids, MASK, pad_token_id=PAD)
        assert out.shape == ids.shape
        assert torch.equal(
            out,
            torch.tensor(
                [
                    [1, 2, 3, PAD, PAD],
                    [4, 5, PAD, PAD, PAD],
                ]
            ),
        )

    def test_tensor_mask_tokens(self):
        ids = torch.tensor([[1, 8, 2, 9, 3]])
        out = remove_tokens(
            ids,
            torch.tensor([8, 9]),
            pad_token_id=PAD,
        )
        assert torch.equal(out, torch.tensor([[1, 2, 3, PAD, PAD]]))

    def test_all_true_mask_is_identity(self):
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        non_mask = torch.ones_like(ids, dtype=torch.bool)
        out = _remove_tokens(ids, non_mask, pad_token_id=PAD)
        assert torch.equal(out, ids)

    def test_all_false_mask_returns_pads(self):
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        non_mask = torch.zeros_like(ids, dtype=torch.bool)
        out = _remove_tokens(ids, non_mask, pad_token_id=PAD)
        assert torch.equal(out, torch.full_like(ids, PAD))


class TestLogSoftmaxLastTwoDims:
    def test_normalises_to_one(self):
        x = torch.randn(2, 3, 4)
        out = log_softmax_last_two_dims(x)
        # exp(out).sum over last two dims must equal 1 per batch element.
        s = out.exp().sum(dim=(1, 2))
        assert torch.allclose(s, torch.ones(2), atol=1e-5)

    def test_gradient_flows(self):
        x = torch.randn(2, 3, 4, requires_grad=True)
        out = log_softmax_last_two_dims(x)
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestMaskedCeLastTwoDims:
    def test_matches_reference_when_no_mask(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 5)
        # Build a valid probability target: one-hot per (batch, seq*vocab).
        target = torch.zeros(2, 3, dtype=torch.long)
        # Convert to flat one-hot of shape (2, 15).
        one_hot = torch.zeros(2, 15)
        for b in range(2):
            one_hot[b, target[b, 0]] = 1.0
        mask = torch.zeros_like(logits, dtype=torch.bool)
        out = masked_ce_last_two_dims(
            logits, one_hot, mask, min_value=-1e9, inplace=False
        )
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(2, -1), one_hot, reduction="none"
        )
        assert torch.allclose(out, expected, atol=1e-5)

    def test_inplace_mutates_logits(self):
        logits = torch.randn(2, 3, 5)
        target = torch.zeros(2, 15)
        target[:, 0] = 1.0
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[0, 0, 0] = True  # mask one position

        original = logits.clone()
        masked_ce_last_two_dims(
            logits, target, mask, min_value=-1e9, inplace=True
        )
        # The masked entry was filled with min_value.
        assert logits[0, 0, 0].item() == -1e9
        # Non-masked entries unchanged.
        assert torch.equal(logits[1], original[1])

    def test_no_inplace_keeps_logits(self):
        logits = torch.randn(2, 3, 5)
        original = logits.clone()
        target = torch.zeros(2, 15)
        target[:, 0] = 1.0
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[0, 0, 0] = True
        masked_ce_last_two_dims(
            logits, target, mask, min_value=-1e9, inplace=False
        )
        assert torch.equal(logits, original)


class TestTopkOverLastTwoDims:
    def test_descending_values(self):
        x = torch.tensor(
            [
                [[1.0, 9.0, 2.0], [3.0, 8.0, 4.0]],
                [[7.0, 6.0, 5.0], [0.0, -1.0, 10.0]],
            ]
        )
        vals, idx = topk_over_last_two_dims(x, k=3)
        assert vals.shape == (2, 3)
        # Values must be sorted descending.
        assert (vals[:, :-1] >= vals[:, 1:]).all()
        # Indices must point back to those values.
        for b in range(2):
            for j in range(3):
                d1, d2 = idx[b, j].tolist()
                assert x[b, d1, d2] == vals[b, j]


class TestMaxOverLastTwoDims:
    def test_index_points_to_max(self):
        x = torch.tensor(
            [
                [[1.0, 9.0, 2.0], [3.0, 8.0, 4.0]],
                [[7.0, 6.0, 5.0], [0.0, -1.0, 10.0]],
            ]
        )
        max_vals, (i1, i2) = max_over_last_two_dims(x)
        assert max_vals.shape == (2,)
        assert torch.allclose(max_vals, torch.tensor([9.0, 10.0]))
        for b in range(2):
            assert x[b, i1[b], i2[b]] == max_vals[b]


class TestSampleOverLastTwoDims:
    def test_argmax_sampling_function_picks_max(self):
        x = torch.tensor(
            [
                [[0.0, 9.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
            ]
        )
        i1, i2 = sample_over_last_two_dims(
            x, sampling_function=lambda flat: flat.argmax(dim=-1)
        )
        assert i1.tolist() == [0, 1]
        assert i2.tolist() == [1, 2]


class TestGeneralSampleOverLastTwoDims:
    def test_falls_back_to_joint_when_second_is_none(self):
        x = torch.tensor(
            [
                [[0.0, 9.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
            ]
        )
        a = general_sample_over_last_two_dims(
            x,
            sampling_function=lambda flat: flat.argmax(dim=-1),
            second_sampling_function=None,
        )
        b = sample_over_last_two_dims(
            x, sampling_function=lambda flat: flat.argmax(dim=-1)
        )
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])

    def test_two_step_sampling_with_argmax_helpers(self):
        # Make a clearly peaked distribution: max is at (seq=1, vocab=2) for
        # batch 0 and (seq=0, vocab=1) for batch 1.
        logits = torch.full((2, 2, 3), -10.0)
        logits[0, 1, 2] = 10.0
        logits[1, 0, 1] = 10.0
        seq_idx, vocab_idx = general_sample_over_last_two_dims(
            logits,
            sampling_function=lambda l: l.argmax(dim=-1),
            second_sampling_function=lambda l: l.argmax(dim=-1),
        )
        assert seq_idx.tolist() == [1, 0]
        assert vocab_idx.tolist() == [2, 1]
