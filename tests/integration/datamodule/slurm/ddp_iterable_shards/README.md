# `ddp_iterable_shards` SLURM scenario

Real multi-GPU `DatasetManager` correctness check for the iterable +
DDP path.  Run on a SLURM cluster as part of the integration suite:

```bash
pytest -m "integration and slurm" tests/integration/datamodule/
```

The test calls `tests.integration._slurm.submit_sbatch_and_wait`,
which invokes:

```text
sbatch --wait script.sh <result_dir> <config_json>
```

`sbatch --wait` blocks until the job is in a terminal state, so the
parent pytest process receives the exit status and per-rank
`rank_<RANK>.json` files atomically.

## Files

| File | Role |
| --- | --- |
| `script.sh` | SLURM batch script.  Sets `MASTER_ADDR`/`MASTER_PORT` from the SLURM allocation and `srun`s `script.py` once per task. Resource directives (`--ntasks=2`, `--gres=gpu:2`, `--time=10:00`, ...) are intentionally minimal so the job fits in most debug QoS. |
| `script.py` | Per-rank entrypoint.  Initialises a NCCL process group, monkey-patches `DatasetManager._download` to read from the in-memory registry, builds an iterable + DDP `DatasetManager`, iterates one epoch, and writes `<result_dir>/rank_<RANK>.json`. |
| `README.md` | This file. |

## Result-file schema

Every rank writes one `rank_<RANK>.json`:

```json
{
  "rank": 0,
  "world_size": 2,
  "ok": true,
  "error": null,
  "backend": "nccl",
  "is_iterable_dataset": true,
  "epochs": [
    {"epoch": 0, "ids": [...], "batch_shapes": [[bs, sl], ...]}
  ]
}
```

On any uncaught exception inside `script.py`, the file is still
written with `ok=false` and the traceback in `error`, so the parent
pytest assertion is always self-contained.

## How to add a new SLURM scenario

1. Create a new sibling directory:
   `tests/integration/datamodule/slurm/<scenario_name>/`
2. Add `script.py`, `script.sh`, `README.md` mirroring the layout
   here.  Reuse `tests.datamodule_helpers` for in-memory datasets
   and the patched `_download`.
3. Add a `slurm`-marked pytest in
   `tests/integration/datamodule/test_dataset_manager_ddp_slurm.py`
   that calls `submit_sbatch_and_wait` with the new `script.sh`
   and asserts on the parsed results.
