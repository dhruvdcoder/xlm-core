from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Optional

import torch


def _infer_batch_len(v: Any, *, dim: int) -> Optional[int]:
    shape = getattr(v, "shape", None)
    if shape is not None:
        try:
            return int(shape[dim])
        except Exception:
            return None
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
        try:
            return len(v)
        except Exception:
            return None
    return None


def _take(v: Any, i: int, *, dim: int) -> Any:
    if dim == 0:
        return v[i]
    shape = getattr(v, "shape", None)
    if shape is not None:
        idx = [slice(None)] * len(shape)
        idx[dim] = i
        return v[tuple(idx)]
    return v[i]


def iter_unbatch(
    batch: Mapping[str, Any],
    length: int,
    *,
    dim: int = 0,
    strict: bool = True,
    broadcast_non_sliceable: bool = False,
) -> Iterator[dict[str, Any]]:
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
    return list(
        iter_unbatch(
            batch,
            length,
            dim=dim,
            strict=strict,
            broadcast_non_sliceable=broadcast_non_sliceable,
        )
    )
