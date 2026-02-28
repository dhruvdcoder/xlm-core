from __future__ import annotations

from collections.abc import Mapping, Sequence, Iterator
from typing import Any, Optional
import torch


def _infer_batch_len(v: Any, *, dim: int) -> Optional[int]:
    """Best-effort length inference along `dim` (returns None if unknown)."""
    # PyTorch / NumPy / array-like
    shape = getattr(v, "shape", None)
    if shape is not None:
        try:
            return int(shape[dim])
        except Exception:
            return None

    # Python sequences (list/tuple, etc.). Strings are sequences too, but usually not batch fields.
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
        try:
            return len(v)
        except Exception:
            return None

    return None


def _take(v: Any, i: int, *, dim: int) -> Any:
    """Take the i-th element along `dim` for common sliceable types."""
    # Fast path for Python sequences (and anything that supports scalar indexing).
    if dim == 0:
        return v[i]

    # For array-likes (torch/numpy), do tuple-based indexing along an arbitrary dim.
    shape = getattr(v, "shape", None)
    if shape is not None:
        # Build [:, :, i, :, :] style index
        idx = [slice(None)] * len(shape)
        idx[dim] = i
        return v[tuple(idx)]

    # Fallback: try direct indexing anyway (may work for custom containers)
    return v[i]


def iter_unbatch(
    batch: Mapping[str, Any],
    length: int,
    *,
    dim: int = 0,
    strict: bool = True,
    broadcast_non_sliceable: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    Convert a "batched" dict into an iterator of per-item dicts.

    Parameters
    ----------
    batch:
        Mapping from field name -> batched value (tensor/ndarray/list/etc.).
    length:
        Number of items to unbatch (user-provided).
    dim:
        Which dimension is the batch dimension for array-likes (torch/numpy).
        For Python lists/tuples, only dim=0 makes sense.
    strict:
        If True, validate (when possible) that each value has batch length == `length`.
    broadcast_non_sliceable:
        If True, values that cannot be indexed are copied as-is into every item.

    Yields
    ------
    dict[str, Any]
        One dict per item in the batch.
    """
    if strict:
        for k, v in batch.items():
            n = _infer_batch_len(v, dim=dim)
            if n is not None and n != length:
                raise ValueError(
                    f"Field {k!r} has batch length {n} along dim={dim}, expected {length}."
                )

    for i in range(length):
        out_i: dict[str, Any] = {}
        for k, v in batch.items():
            try:
                out_i[k] = _take(v, i, dim=dim)
                if isinstance(out_i[k], torch.Tensor):
                    out_i[k] = out_i[k].tolist()
            except Exception:
                if broadcast_non_sliceable:
                    out_i[k] = v
                else:
                    raise TypeError(
                        f"Field {k!r} of type {type(v).__name__} could not be indexed at i={i} "
                        f"(dim={dim}). Consider setting broadcast_non_sliceable=True."
                    )
        yield out_i


def unbatch(
    batch: Mapping[str, Any],
    length: int,
    *,
    dim: int = 0,
    strict: bool = False,
    broadcast_non_sliceable: bool = True,
) -> list[dict[str, Any]]:
    """List-returning wrapper around `iter_unbatch`."""
    return list(
        iter_unbatch(
            batch,
            length,
            dim=dim,
            strict=strict,
            broadcast_non_sliceable=broadcast_non_sliceable,
        )
    )
