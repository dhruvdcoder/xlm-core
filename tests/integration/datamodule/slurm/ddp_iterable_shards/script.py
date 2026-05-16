"""Multi-GPU DDP entrypoint for the ``slurm`` ``ddp_iterable_shards`` scenario.

This script is launched once per rank by ``srun`` (see ``script.sh``).
Each rank:

1. Initialises a NCCL process group from the SLURM-set environment.
2. Monkey-patches :meth:`xlm.datamodule.DatasetManager._download` to
   read from the in-memory dataset registry (no network I/O on the
   compute node).
3. Builds a :class:`DatasetManager` configured for iterable + DDP and
   iterates the dataloader for one epoch.
4. Writes ``<result_dir>/rank_<RANK>.json`` describing what it observed.

Same JSON result schema as
:mod:`tests.integration._scripts.ddp_dsm_entrypoint`, so the parsing
logic in :func:`tests.integration._slurm.submit_sbatch_and_wait`
mirrors :func:`tests.integration._runner.run_cpu_distributed`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--config-json", type=str, default="{}")
    return parser.parse_args()


def _read_dist_env() -> Dict[str, int]:
    return {
        "rank": int(os.environ["RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
    }


def _make_tokenizer():
    from xlm.datamodule import SimpleSpaceTokenizer

    return SimpleSpaceTokenizer(vocab=[str(i) for i in range(50)])


def _run_one_rank(
    rank: int,
    world_size: int,
    local_rank: int,
    result_dir: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    import torch
    import torch.distributed as dist

    from tests.datamodule_helpers import (
        EXAMPLE_TOKEN_LEN,
        IdTrackingCollator,
        build_inmem_datasets,
        patch_dataset_manager_download,
    )
    from xlm.datamodule import DatasetManager

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

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
    dsm_kwargs.setdefault("iterable_dataset_shards", 4)
    dsm_kwargs.setdefault(
        "dataloader_kwargs",
        {"batch_size": 4, "num_workers": 1, "pin_memory": False},
    )
    dsm_kwargs.setdefault("use_manual_cache", False)

    dsm = DatasetManager(**dsm_kwargs)

    cache_dir = result_dir / f"_dsm_cache_rank_{rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dsm.setup(
        stage="fit",
        manual_cache_dir=str(cache_dir),
        tokenizer=tokenizer,
        block_size=EXAMPLE_TOKEN_LEN,
        is_ddp=True,
        rank=rank,
        world_size=world_size,
    )

    dl = dsm.get_dataloader(
        type="train", is_ddp=True, rank=rank, world_size=world_size
    )

    run_cfg: Dict[str, Any] = dict(config.get("run") or {})
    max_batches: int = int(run_cfg.get("max_batches_per_epoch", 1000))
    ids: List[int] = []
    shapes: List[List[int]] = []
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        ids.extend(int(x) for x in batch["ids"])
        shapes.append(list(batch["input_ids"].shape))

    dist.barrier()

    return {
        "rank": rank,
        "world_size": world_size,
        "ok": True,
        "error": None,
        "backend": backend,
        "is_iterable_dataset": bool(dsm.is_iterable_dataset),
        "epochs": [{"epoch": 0, "ids": ids, "batch_shapes": shapes}],
    }


def _write_result(result_dir: Path, rank: int, payload: Dict[str, Any]) -> None:
    path = result_dir / f"rank_{rank}.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f)
    tmp.replace(path)


def main() -> int:
    args = _parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(
        os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1))
    )
    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0))
    )

    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        _write_result(
            args.result_dir,
            rank,
            {
                "rank": rank,
                "world_size": world_size,
                "ok": False,
                "error": f"Invalid --config-json: {exc}",
                "epochs": [],
            },
        )
        return 2

    try:
        payload = _run_one_rank(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            result_dir=args.result_dir,
            config=config,
        )
    except Exception:
        payload = {
            "rank": rank,
            "world_size": world_size,
            "ok": False,
            "error": traceback.format_exc(),
            "epochs": [],
        }
    finally:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    _write_result(args.result_dir, rank, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
