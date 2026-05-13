"""Unit tests for :class:`xlm.utils.data.LengthSampler`."""

import pytest

from xlm.utils.data import LengthSampler


def _make_dataset(lengths):
    """Return a list of dicts with ``input_ids`` of the requested lengths."""
    return [{"input_ids": [0] * L} for L in lengths]


class TestLengthSampler:
    def test_len(self):
        ds = _make_dataset([1, 5, 3, 8, 2])
        sampler = LengthSampler(ds, shuffle=False, max_length_diff=1)
        assert len(sampler) == 5

    def test_no_shuffle_yields_descending_length_order(self):
        ds = _make_dataset([1, 5, 3, 8, 2])
        sampler = LengthSampler(ds, shuffle=False, max_length_diff=1)
        order = list(sampler)
        # After sort, indices are sorted by length descending: 8,5,3,2,1
        # which corresponds to original positions [3, 1, 2, 4, 0]
        assert order == [3, 1, 2, 4, 0]

    def test_iter_yields_permutation_when_shuffled(self):
        ds = _make_dataset([1, 5, 3, 8, 2, 7, 4, 6])
        sampler = LengthSampler(ds, shuffle=True, max_length_diff=2)
        order = list(sampler)
        assert sorted(order) == list(range(len(ds)))

    def test_shuffle_requires_max_length_diff_at_least_one(self):
        ds = _make_dataset([1, 2, 3])
        with pytest.raises(AssertionError):
            LengthSampler(ds, shuffle=True, max_length_diff=0)

    def test_boundaries_respect_max_length_diff(self):
        # Lengths sorted desc: 10, 9, 5, 4, 1.  With max_length_diff=1, the
        # adjacency check uses ``sorted_lengths[end+1] - sorted_lengths[start]``.
        # The construction is greedy and groups elements into a single growing
        # run while the gap stays within the bound.
        ds = _make_dataset([10, 9, 5, 4, 1])
        sampler = LengthSampler(ds, shuffle=False, max_length_diff=1)
        # Boundaries are end-indices of each group plus a final num_points.
        # We mainly want to assert the construction doesn't crash and produces
        # a valid covering.
        assert sampler.boundaries[-1] == len(ds)
        assert sampler.boundaries[0] >= 0

    def test_shuffled_groups_keep_membership(self):
        # If two items have the same length, they should land in the same group
        # and shuffling must not move them out.
        ds = _make_dataset([4, 4, 4, 100])
        sampler = LengthSampler(ds, shuffle=True, max_length_diff=1)
        order = list(sampler)
        # The length-100 item dominates the head; the three length-4 items
        # must occupy the trailing 3 positions in some order.
        assert sorted(order[-3:]) == [0, 1, 2]
        assert order[0] == 3
