"""Unit tests for :mod:`mlm.unbatch`."""

import pytest
import torch

from mlm.unbatch import iter_unbatch, unbatch


class TestIterUnbatch:
    def test_basic_dim0_tensor(self):
        batch = {"x": torch.arange(6).view(3, 2)}
        items = list(iter_unbatch(batch, length=3))
        assert len(items) == 3
        # Tensors are converted to nested lists.
        assert items[0]["x"] == [0, 1]
        assert items[1]["x"] == [2, 3]
        assert items[2]["x"] == [4, 5]

    def test_multiple_fields(self):
        batch = {
            "x": torch.arange(6).view(3, 2),
            "y": ["a", "b", "c"],
        }
        items = list(iter_unbatch(batch, length=3))
        assert items[0]["y"] == "a"
        assert items[1]["y"] == "b"
        assert items[2]["y"] == "c"

    def test_dim1_indexing(self):
        # Take along axis 1.
        batch = {"x": torch.arange(6).view(2, 3)}
        items = list(iter_unbatch(batch, length=3, dim=1))
        assert len(items) == 3
        assert items[0]["x"] == [0, 3]
        assert items[1]["x"] == [1, 4]
        assert items[2]["x"] == [2, 5]

    def test_strict_validation_catches_mismatched_lengths(self):
        batch = {
            "x": torch.arange(6).view(3, 2),
            "y": ["a", "b"],  # wrong length
        }
        with pytest.raises(ValueError, match="batch length"):
            list(iter_unbatch(batch, length=3, strict=True))

    def test_non_strict_skips_validation(self):
        batch = {
            "x": torch.arange(6).view(3, 2),
            "y": ["a", "b"],
        }
        # With strict=False the length mismatch is no longer a ValueError;
        # the IndexError ultimately surfaces as a TypeError (since indexing
        # ``y[2]`` blows up and broadcast is off by default).
        with pytest.raises((IndexError, TypeError)):
            list(iter_unbatch(batch, length=3, strict=False))

    def test_broadcast_non_sliceable_with_scalar(self):
        # A bare int is not indexable; with broadcast it is copied verbatim.
        batch = {"x": torch.arange(4).view(2, 2), "k": 5}
        items = list(
            iter_unbatch(
                batch, length=2, strict=False, broadcast_non_sliceable=True
            )
        )
        assert items[0]["k"] == 5 and items[1]["k"] == 5

    def test_no_broadcast_raises(self):
        batch = {"x": torch.arange(4).view(2, 2), "k": 5}
        with pytest.raises(TypeError, match="could not be indexed"):
            list(
                iter_unbatch(
                    batch,
                    length=2,
                    strict=False,
                    broadcast_non_sliceable=False,
                )
            )

    def test_string_field_skips_strict_length_check(self):
        # "ab" has length 2 but ``_infer_batch_len`` excludes strings entirely,
        # so a strict run does not fault it for "wrong length". The string is
        # then character-indexed by ``_take`` (this is the actual behaviour;
        # we only assert the strict pre-check is skipped).
        batch = {"x": torch.arange(4).view(2, 2), "label": "ab"}
        items = list(iter_unbatch(batch, length=2, strict=True))
        assert len(items) == 2
        # Per-row characters from the indexed string.
        assert items[0]["label"] == "a"
        assert items[1]["label"] == "b"


class TestUnbatch:
    def test_returns_list(self):
        batch = {"x": torch.arange(6).view(3, 2)}
        out = unbatch(batch, length=3)
        assert isinstance(out, list)
        assert len(out) == 3
