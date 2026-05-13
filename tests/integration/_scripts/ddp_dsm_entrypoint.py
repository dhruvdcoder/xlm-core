"""Subprocess entrypoint for CPU multi-rank ``DatasetManager`` integration tests.

Launched by :func:`tests.integration._runner.run_cpu_distributed` under
``torch.distributed.run`` with ``--nproc_per_node=<world_size>``.  Each
spawned process:

1. Reads ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` from the environment
   (set by torchrun) and initialises a process group on the ``gloo``
   backend (CPU-friendly, available on every PyTorch install).
2. Monkey-patches :meth:`xlm.datamodule.DatasetManager._download` so the
   manager reads from the in-memory dataset registry built by
   :func:`tests.datamodule_helpers.build_inmem_datasets`; no network
   I/O ever happens.
3. Builds a :class:`DatasetManager` from a JSON-encoded config, calls
   ``setup`` and ``get_dataloader`` for the train split, and iterates
   the loader for one or more epochs, recording the example ``id`` of
   every batch element seen by this rank.
4. Writes ``rank_<RANK>.json`` into ``--result-dir`` with the per-epoch
   coverage, batch shapes, and any captured exception so the parent
   pytest process can make per-rank assertions.

Config schema (JSON, passed via ``--config-json``)
==================================================

::

    {
      "dsm_kwargs": {                    # forwarded to DatasetManager(**...)
        "full_name": "mem/raw_large/train",
        "iterable_dataset_shards": 4,
        ...
      },
      "setup_kwargs": {                  # forwarded to dsm.setup(**...)
        "block_size": 6,                 # default: EXAMPLE_TOKEN_LEN
        "stage": "fit"
      },
      "run": {
        "num_epochs": 1,                 # default 1
        "max_batches_per_epoch": 1000,   # safety cap, default 1000
        "set_epoch_each_epoch": true     # iterable: dsm.set_epoch / map: sampler.set_epoch
      }
    }

The result file mirrors this::

    {
      "rank": 0,
      "world_size": 2,
      "ok": true,
      "error": null,
      "is_iterable_dataset": true,
      "epochs": [
        {"epoch": 0, "ids": [...], "batch_shapes": [[bs, sl], ...]}
      ]
    }

Anything that goes wrong (import failure, init_process_group failure,
DatasetManager raising, etc.) is caught at the outermost level and
written into ``rank_<RANK>.json`` with ``ok=False`` and a textual
traceback so the parent test gets a useful error message.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Directory to write rank_<RANK>.json into.",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default="{}",
        help="JSON-encoded configuration; see module docstring for schema.",
    )
    return parser.parse_args()


def _read_dist_env() -> Dict[str, int]:
    """Read RANK / WORLD_SIZE / LOCAL_RANK from torchrun-set env vars."""
    return {
        "rank": int(os.environ["RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
    }


def _make_tokenizer():
    """Build the same SimpleSpaceTokenizer the pytest fixtures use."""
    from xlm.datamodule import SimpleSpaceTokenizer

    return SimpleSpaceTokenizer(vocab=[str(i) for i in range(50)])


def _run_one_rank(
    rank: int,
    world_size: int,
    result_dir: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Body of the subprocess: build & iterate a DatasetManager.

    Returns the dict written to ``rank_<rank>.json``.  Exceptions are
    caught by the caller and serialised into the same dict shape with
    ``ok=False`` and ``error=<traceback string>``.
    """
    import torch
    import torch.distributed as dist

    from tests.datamodule_helpers import (
        EXAMPLE_TOKEN_LEN,
        IdTrackingCollator,
        build_inmem_datasets,
        patch_dataset_manager_download,
    )
    from xlm.datamodule import DatasetManager

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

    # ``DatasetManager.get_dataloader`` pulls a fresh seed from the *global*
    # torch RNG for ``StatefulDistributedSampler`` (case 2).  Production code
    # relies on ``lightning.seed_everything`` having been called *before* the
    # data module so every rank lands on the same RNG state and therefore
    # the same sampler seed.  The runner is not a Lightning Trainer, so we
    # mimic that behaviour explicitly here using a config-supplied seed.
    seed = int((config.get("run") or {}).get("seed", 0))
    torch.manual_seed(seed)

    patch_dataset_manager_download(build_inmem_datasets())

    tokenizer = _make_tokenizer()
    collator = IdTrackingCollator(tokenizer=tokenizer)

    dsm_kwargs: Dict[str, Any] = dict(config.get("dsm_kwargs") or {})
    dsm_kwargs.setdefault("full_name", "mem/raw_large/train")
    dsm_kwargs.setdefault("full_name_debug", dsm_kwargs["full_name"])
    dsm_kwargs.setdefault("collator", collator)
    dsm_kwargs.setdefault(
        "preprocess_function",
        "tests.datamodule_helpers.example_to_input_ids",
    )
    dsm_kwargs.setdefault("columns_to_keep", ["id"])
    dsm_kwargs.setdefault(
        "dataloader_kwargs",
        {"batch_size": 2, "num_workers": 0, "pin_memory": False},
    )
    dsm_kwargs.setdefault("use_manual_cache", False)

    dsm = DatasetManager(**dsm_kwargs)

    setup_kwargs: Dict[str, Any] = dict(config.get("setup_kwargs") or {})
    setup_kwargs.setdefault("stage", "fit")
    setup_kwargs.setdefault("block_size", EXAMPLE_TOKEN_LEN)
    manual_cache_dir = setup_kwargs.pop("manual_cache_dir", None)
    if manual_cache_dir is None:
        manual_cache_dir = str(result_dir / f"_dsm_cache_rank_{rank}")
        Path(manual_cache_dir).mkdir(parents=True, exist_ok=True)

    dsm.setup(
        manual_cache_dir=manual_cache_dir,
        tokenizer=tokenizer,
        is_ddp=True,
        rank=rank,
        world_size=world_size,
        **setup_kwargs,
    )

    is_iterable = bool(dsm.is_iterable_dataset)

    dl = dsm.get_dataloader(
        type="train",
        is_ddp=True,
        rank=rank,
        world_size=world_size,
    )

    run_cfg: Dict[str, Any] = dict(config.get("run") or {})
    num_epochs: int = int(run_cfg.get("num_epochs", 1))
    max_batches_per_epoch: int = int(run_cfg.get("max_batches_per_epoch", 1000))
    set_epoch_each_epoch: bool = bool(run_cfg.get("set_epoch_each_epoch", True))

    epochs_log: List[Dict[str, Any]] = []
    for epoch in range(num_epochs):
        if set_epoch_each_epoch:
            if is_iterable:
                dsm.set_epoch(epoch)
            else:
                # Map-style DDP path attaches a StatefulDistributedSampler.
                sampler = getattr(dl, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

        ids: List[int] = []
        shapes: List[List[int]] = []
        for i, batch in enumerate(dl):
            if i >= max_batches_per_epoch:
                break
            ids.extend(int(x) for x in batch["ids"])
            shapes.append(list(batch["input_ids"].shape))
        epochs_log.append(
            {"epoch": epoch, "ids": ids, "batch_shapes": shapes}
        )

    dist.barrier()

    return {
        "rank": rank,
        "world_size": world_size,
        "ok": True,
        "error": None,
        "is_iterable_dataset": is_iterable,
        "epochs": epochs_log,
    }


def _write_result(result_dir: Path, rank: int, payload: Dict[str, Any]) -> None:
    path = result_dir / f"rank_{rank}.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f)
    # Atomic from the parent's POV, even if multiple ranks finish nearly
    # simultaneously.
    tmp.replace(path)


def main() -> int:
    args = _parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        # We do not have RANK at this point if torchrun didn't set the env
        # for some reason; default to 0 so the parent gets *something*.
        rank = int(os.environ.get("RANK", 0))
        _write_result(
            args.result_dir,
            rank,
            {
                "rank": rank,
                "world_size": int(os.environ.get("WORLD_SIZE", 1)),
                "ok": False,
                "error": f"Invalid --config-json: {exc}",
                "is_iterable_dataset": None,
                "epochs": [],
            },
        )
        return 2

    dist_env = _read_dist_env()

    payload: Optional[Dict[str, Any]] = None
    try:
        payload = _run_one_rank(
            rank=dist_env["rank"],
            world_size=dist_env["world_size"],
            result_dir=args.result_dir,
            config=config,
        )
    except Exception:
        payload = {
            "rank": dist_env["rank"],
            "world_size": dist_env["world_size"],
            "ok": False,
            "error": traceback.format_exc(),
            "is_iterable_dataset": None,
            "epochs": [],
        }
    finally:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    _write_result(args.result_dir, dist_env["rank"], payload)
    # Always return 0 so torchrun does not flag a "successful" run
    # (with per-rank ok=False) as a hard subprocess failure -- the
    # parent runner already consumes ok / error from the JSON files.
    return 0


if __name__ == "__main__":
    sys.exit(main())
