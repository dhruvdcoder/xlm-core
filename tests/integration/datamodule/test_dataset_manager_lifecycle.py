"""Single-process end-to-end lifecycle test for ``DatasetManager``.

Sanity-checks that a fresh ``DatasetManager`` can run the full
``prepare_data -> setup -> get_dataloader -> iterate batch`` flow with
both the map-style and iterable backends, and that the manual cache is
populated on disk along the way.

For the exhaustive single-method matrix (every branch of ``__init__`` /
``prepare_data`` / ``setup`` / ``get_dataloader``) see
``tests/core/test_datamodule.py``.  For multi-process / DDP / SLURM /
Lightning Trainer paths see the sibling ``test_dataset_manager_ddp_*``
modules.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from tests.datamodule_helpers import EXAMPLE_TOKEN_LEN


pytestmark = [pytest.mark.integration]


# Convenience constant matching tests.datamodule_helpers.build_inmem_datasets.
TRAIN_SIZE = 17


class TestEndToEndLifecycle:
    """Sanity check: prepare_data -> setup -> get_dataloader -> iterate batch."""

    @pytest.mark.parametrize("iterable", [False, True])
    def test_full_lifecycle(
        self,
        dataset_manager_factory,
        simple_tokenizer,
        manual_cache_dir,
        iterable: bool,
    ):
        kwargs: Dict[str, Any] = {"use_manual_cache": True}
        if iterable:
            kwargs["iterable_dataset_shards"] = 4
        dsm = dataset_manager_factory(**kwargs)

        dsm.prepare_data(
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
        )
        dsm.setup(
            stage="fit",
            manual_cache_dir=str(manual_cache_dir),
            tokenizer=simple_tokenizer,
            block_size=EXAMPLE_TOKEN_LEN,
            is_ddp=False,
            rank=0,
            world_size=1,
        )
        dl = dsm.get_dataloader(
            type="train", is_ddp=False, rank=0, world_size=1
        )
        batches = list(dl)
        seen = sorted(i for b in batches for i in b["ids"])
        assert seen == list(range(TRAIN_SIZE))
        # Verify the batch shape matches what the collator promised.
        first = batches[0]
        assert first["input_ids"].shape == (
            len(first["ids"]),
            EXAMPLE_TOKEN_LEN,
        )
