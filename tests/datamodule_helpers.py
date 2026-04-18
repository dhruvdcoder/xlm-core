"""Importable helpers for ``DatasetManager`` tests (unit and integration).

Lives at ``tests/datamodule_helpers.py`` -- one level above
``tests/integration/`` -- so both the fast single-method tests in
``tests/core/test_datamodule.py`` and the multi-process / SLURM
entrypoints under ``tests/integration/`` can share the same in-memory
datasets, monkey-patch helpers, processors, and collator.

Everything in this module must be importable from arbitrary subprocesses
(e.g. the multi-process DDP runner and the SLURM scripts), so it must NOT
depend on any pytest fixtures or pytest-only context.

Provided utilities:

* :func:`build_inmem_datasets` -- constructs a stable, in-memory
  ``{full_name: datasets.Dataset}`` registry used to monkey-patch
  :meth:`xlm.datamodule.DatasetManager._download`.
* :func:`make_patched_download` -- builds the replacement ``_download``
  function bound to a given registry.  Used by the pytest fixture together
  with ``monkeypatch.setattr`` for automatic restoration.
* :func:`patch_dataset_manager_download` -- standalone monkey-patch helper
  (no pytest dependency).  Applies the patch immediately and returns a
  thunk that restores the original ``_download`` method.  Use this from
  subprocess entrypoints that have no access to pytest fixtures.
* :func:`example_to_input_ids` -- preprocess function compatible with
  :meth:`xlm.datamodule.DatasetManager._preprocess`.
* :func:`pack_with_id` -- group processor used by the iterable group-
  processor path; passes through the ``id`` field for coverage tracking.
* :func:`drop_first_example` -- on-the-fly filter function for testing
  :attr:`xlm.datamodule.DatasetManager.on_the_fly_filter_fn`.
* :func:`identity_processor` -- on-the-fly per-example processor.
* :class:`IdTrackingCollator` -- minimal collator that stacks fixed-length
  ``input_ids``/``attention_mask``/``token_type_ids`` into tensors and
  carries the ``id`` column through as a list, so DDP coverage tests can
  see exactly which example each rank consumed.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import datasets
import torch


# ---------------------------------------------------------------------------
# In-memory datasets registry
# ---------------------------------------------------------------------------

#: Fixed token-id length for every example in the in-memory datasets.
#: Keeping it constant means the collator does not need to pad.
EXAMPLE_TOKEN_LEN = 6

#: Vocabulary id range for the tokens.  Matches ``SimpleSpaceTokenizer``
#: built from ``["0".."49"]`` -- vocab tokens occupy ids 7..56.
_TOKEN_ID_LO = 7
_TOKEN_ID_HI = 56


def _make_token_ids(rng: random.Random, n: int = EXAMPLE_TOKEN_LEN) -> List[int]:
    return [rng.randint(_TOKEN_ID_LO, _TOKEN_ID_HI) for _ in range(n)]


def _build_split(
    seed: int,
    n: int,
    id_offset: int,
) -> datasets.Dataset:
    """Build an in-memory split with deterministic content.

    Each row has ``id`` (globally unique across splits via ``id_offset``),
    ``text`` (a placeholder string), and ``token_ids`` (length
    :data:`EXAMPLE_TOKEN_LEN`, values in the SimpleSpaceTokenizer range).
    """
    rng = random.Random(seed)
    rows = {
        "id": [id_offset + i for i in range(n)],
        "text": [f"row_{id_offset + i}" for i in range(n)],
        "token_ids": [_make_token_ids(rng) for _ in range(n)],
    }
    return datasets.Dataset.from_dict(rows)


#: Per-shard ``len(token_ids)`` plan for ``mem/raw_varlen/train``.
#:
#: Designed so :func:`xlm.datamodule.pack_sequences` (with
#: ``use_bos=True``, ``drop_last=True``, ``block_size=8``) yields
#: **3 / 4 / 3 / 6** blocks per contiguous shard when the dataset is
#: converted via ``Dataset.to_iterable_dataset(num_shards=4)``.
#:
#: Per-example contribution to the packed stream is ``bos + tokens + eos
#: = len + 2``.  Stream totals per shard:
#:   shard 0: 4+4+4+4 + 4*2 = 24 -> 24 // 8 = 3 blocks
#:   shard 1: 6+6+6+6 + 4*2 = 32 -> 32 // 8 = 4 blocks
#:   shard 2: 4+4+4+4 + 4*2 = 24 -> 24 // 8 = 3 blocks
#:   shard 3: 10+10+10+10 + 4*2 = 48 -> 48 // 8 = 6 blocks
#:
#: Under ``split_dataset_by_node(world_size=2)`` (contiguous shard
#: distribution) the per-rank totals are therefore **7 vs 9 batches**
#: at ``batch_size=1`` -- the canonical "uneven shards hang DDP"
#: pathology.  See
#: ``tests/integration/datamodule/test_dataset_manager_ddp_cpu.py``
#: ``TestIterableDdpPackSequencesUnevenBatches``.
VARLEN_SHARD_LENGTHS: List[List[int]] = [
    [4, 4, 4, 4],
    [6, 6, 6, 6],
    [4, 4, 4, 4],
    [10, 10, 10, 10],
]


def _build_varlen_split(
    seed: int,
    shard_lengths: Sequence[Sequence[int]],
    id_offset: int,
) -> datasets.Dataset:
    """Build an in-memory split with **variable** ``token_ids`` lengths.

    Rows are emitted shard by shard in the order of ``shard_lengths`` so
    that ``Dataset.to_iterable_dataset(num_shards=len(shard_lengths))``
    -- which chunks contiguously -- places each ``shard_lengths[i]``
    group into shard ``i``.  This gives the test deterministic control
    over per-shard token counts (and therefore over the number of
    packed blocks each shard produces under
    :func:`xlm.datamodule.pack_sequences`).
    """
    rng = random.Random(seed)
    rows: Dict[str, List[Any]] = {"id": [], "text": [], "token_ids": []}
    next_id = id_offset
    for shard in shard_lengths:
        for ln in shard:
            rows["id"].append(next_id)
            rows["text"].append(f"row_{next_id}")
            rows["token_ids"].append(
                [rng.randint(_TOKEN_ID_LO, _TOKEN_ID_HI) for _ in range(ln)]
            )
            next_id += 1
    return datasets.Dataset.from_dict(rows)


def build_inmem_datasets() -> Dict[str, datasets.Dataset]:
    """Build the canonical in-memory dataset registry.

    Keys follow the same ``"<repo>/<ds_name>/<split>"`` convention used by
    :class:`xlm.datamodule.DatasetManager`.  Globally unique ``id`` values
    let DDP tests assert per-rank coverage / non-overlap directly.
    """
    return {
        # Mirrors the 17/7/5 sizes that HF's test_distributed.py uses so
        # split-by-node arithmetic is easy to reason about.
        "mem/raw/train": _build_split(seed=0, n=17, id_offset=0),
        "mem/raw/val": _build_split(seed=1, n=7, id_offset=100),
        "mem/raw/test": _build_split(seed=2, n=5, id_offset=200),
        # 60 examples = enough for to_iterable_dataset(num_shards=4) and
        # multi-rank * multi-worker tests.
        "mem/raw_large/train": _build_split(seed=3, n=60, id_offset=1000),
        # Variable-length sequences laid out so contiguous sharding by
        # ``to_iterable_dataset(num_shards=4)`` gives unequal per-shard
        # token streams; see ``VARLEN_SHARD_LENGTHS`` for the exact plan.
        "mem/raw_varlen/train": _build_varlen_split(
            seed=4,
            shard_lengths=VARLEN_SHARD_LENGTHS,
            id_offset=2000,
        ),
    }


# ---------------------------------------------------------------------------
# DatasetManager._download monkey-patch (importable, no pytest dep)
# ---------------------------------------------------------------------------


def make_patched_download(
    registry: Mapping[str, datasets.Dataset],
) -> Callable[..., datasets.Dataset]:
    """Return a replacement for ``DatasetManager._download`` bound to ``registry``.

    The returned function has the same signature as the original
    (``self, num_proc=None``) and honors :attr:`DatasetManager.train_test_split`
    exactly the way the real method does.  Lookup falls back from
    ``self.full_name`` to ``f"{self.name}/train"`` when train_test_split is
    requested (mirroring the real ``_download``).
    """

    def _download(self, num_proc=None):  # type: ignore[no-redef]
        full = self.full_name
        if full in registry:
            ds = registry[full]
        elif self.train_test_split is not None:
            # Real _download reads the "train" split when train_test_split
            # is requested, regardless of the requested split.
            train_key = f"{self.name}/train"
            if train_key not in registry:
                raise KeyError(
                    f"In-memory dataset not registered: {full!r} "
                    f"(also tried {train_key!r}). "
                    f"Available: {sorted(registry)}"
                )
            ds = registry[train_key]
        else:
            raise KeyError(
                f"In-memory dataset not registered: {full!r}. "
                f"Available: {sorted(registry)}"
            )

        if self.train_test_split is not None:
            ds = ds.train_test_split(
                test_size=self.train_test_split["size"],
                seed=self.train_test_split["seed"],
            )
            try:
                ds = ds[self.train_test_split["split"]]
            except KeyError as exc:  # pragma: no cover -- defensive
                raise ValueError(
                    f"Invalid split to use: {self.train_test_split['split']}"
                ) from exc

        # Return a fresh, fully in-memory copy so cross-test mutations
        # (cache writes, set_format, ...) do not leak between tests.
        return datasets.Dataset.from_dict(ds.to_dict())

    return _download


def patch_dataset_manager_download(
    registry: Mapping[str, datasets.Dataset],
) -> Callable[[], None]:
    """Apply :func:`make_patched_download` to the live ``DatasetManager`` class.

    Returns a callable that restores the original ``_download`` when invoked.
    Designed for subprocess scripts that have no access to pytest fixtures.
    """
    from xlm.datamodule import DatasetManager  # local import for subprocess use

    original = DatasetManager._download
    DatasetManager._download = make_patched_download(registry)

    def _restore() -> None:
        DatasetManager._download = original

    return _restore


# ---------------------------------------------------------------------------
# Preprocess / on-the-fly processor functions
# ---------------------------------------------------------------------------


def example_to_input_ids(
    example: Dict[str, Any],
    tokenizer: Any,
    block_size: Optional[int] = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Per-example preprocess: ``token_ids`` -> input_ids/mask/type_ids.

    Designed to work with ``columns_to_keep=["id"]`` so the ``id`` column
    survives the ``ds.map(remove_columns=...)`` call inside
    :meth:`DatasetManager._preprocess`.
    """
    token_ids = list(example["token_ids"])
    if block_size is not None:
        token_ids = token_ids[:block_size]
    n = len(token_ids)
    return {
        "input_ids": token_ids,
        "attention_mask": [1] * n,
        "token_type_ids": [0] * n,
    }


def identity_processor(
    example: Dict[str, Any],
    tokenizer: Any,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Pass-through on-the-fly per-example processor.

    Exists to exercise the ``on_the_fly_processor`` branch in
    :meth:`DatasetManager._apply_on_the_fly_processors` without altering
    example content (so coverage assertions still work).
    """
    return dict(example)


def drop_first_example(example: Dict[str, Any]) -> bool:
    """On-the-fly filter that drops the row with the smallest ``id`` per batch.

    Used to exercise the ``on_the_fly_filter_fn`` branch.  Filter functions
    receive a single example, so we drop ``id == 0`` specifically.
    """
    return example.get("id", -1) != 0


def pack_with_id(
    examples: Dict[str, List[Any]],
    tokenizer: Any,
    block_size: int,
    **_kwargs: Any,
) -> Dict[str, List[Any]]:
    """Group processor that packs fixed-length rows into ``block_size`` blocks.

    Differs from :func:`xlm.datamodule.pack_sequences` in that we:

    * preserve the ``id`` column as a list-of-lists (one inner list per
      output block, recording which input ``id`` values contributed);
    * do not depend on ``tokenizer.eos_token_id``.

    Inputs come in as a *batched* dict: each key maps to a list, one entry
    per example in the batch.
    """
    ids: List[int] = list(examples["id"])
    chunks = [list(ids_chunk) for ids_chunk in examples["input_ids"]]
    masks = [list(m) for m in examples["attention_mask"]]
    types = [list(t) for t in examples["token_type_ids"]]

    flat_ids: List[int] = []
    flat_mask: List[int] = []
    flat_type: List[int] = []
    flat_src: List[int] = []
    for src_id, blk, m, t in zip(ids, chunks, masks, types):
        flat_ids.extend(blk)
        flat_mask.extend(m)
        flat_type.extend(t)
        flat_src.extend([src_id] * len(blk))

    out_input_ids: List[List[int]] = []
    out_attention: List[List[int]] = []
    out_token_type: List[List[int]] = []
    out_source_ids: List[List[int]] = []
    out_id: List[int] = []
    next_id = -(10**6)  # synthetic ids for packed blocks; do not collide w/ raw

    for start in range(0, len(flat_ids) - block_size + 1, block_size):
        end = start + block_size
        out_input_ids.append(flat_ids[start:end])
        out_attention.append(flat_mask[start:end])
        out_token_type.append(flat_type[start:end])
        out_source_ids.append(flat_src[start:end])
        out_id.append(next_id)
        next_id += 1

    return {
        "id": out_id,
        "input_ids": out_input_ids,
        "attention_mask": out_attention,
        "token_type_ids": out_token_type,
        "source_ids": out_source_ids,
    }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


class IdTrackingCollator:
    """Minimal collator that stacks fixed-length tensors and keeps ``id``.

    The in-memory datasets in this suite produce examples of identical
    ``EXAMPLE_TOKEN_LEN`` length, so no padding is needed.  Carrying the
    ``id`` field through as a Python list lets DDP / shuffling tests
    assert exact per-rank coverage.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        block_size: Optional[int] = None,
        noise_schedule: Any = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule

    def __call__(
        self, examples: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        # ``id`` is dropped by group processors that change batch size
        # (e.g. ``xlm.datamodule.pack_sequences``).  Fall back to an empty
        # list so coverage-style assertions can still inspect everything
        # else (``input_ids`` shapes, batch counts, ...).
        has_id = bool(examples) and "id" in examples[0]
        return {
            "input_ids": torch.tensor(
                [list(e["input_ids"]) for e in examples], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [list(e["attention_mask"]) for e in examples], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                [list(e["token_type_ids"]) for e in examples], dtype=torch.long
            ),
            "ids": [int(e["id"]) for e in examples] if has_id else [],
        }
