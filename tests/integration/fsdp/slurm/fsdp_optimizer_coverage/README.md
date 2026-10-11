# `fsdp_optimizer_coverage` SLURM scenario

End-to-end 2-GPU check for [issue #59](https://github.com/dhruvdcoder/xlm-core/issues/59):
under `trainer_strategy=fsdp` with a per-block `auto_wrap_policy`, the root FSDP
unit (embeddings, final norm, output head) must be part of the optimizer, and
`Harness.configure_optimizers` must fail fast when it is not.

The CPU gloo test (`tests/integration/test_fsdp_optimizer_coverage.py`) covers
the `Harness` helpers with raw `FullyShardedDataParallel`. Lightning's
`FSDPStrategy` needs CUDA, so the wrap-then-`configure_optimizers` ordering and
the inherited Hydra config can only be exercised here.

```bash
XLM_INTEGRATION_SLURM_ENABLE=1 \
XLM_INTEGRATION_SBATCH_ARGS="--partition=gpu,--constraint=a100|l40s" \
pytest -m "integration and slurm" tests/integration/fsdp/ -s
```

The test calls `tests.integration._slurm.submit_sbatch_and_wait`, which runs
`sbatch --wait script.sh <result_dir> <config_json>`.

## Files

| File | Role |
| --- | --- |
| `script.sh` | SLURM batch script. Runs `prepare_data` for `star_easy_mlm` once, then `srun`s `script.py` on 2 tasks / 2 GPUs (Lightning's `SLURMEnvironment` derives rank / address / port from the SLURM variables, as in `configs/slurm/train_sbatch.yaml`). Sets `NCCL_P2P_DISABLE=1` (host-memory transport; a node with a broken CUDA P2P path otherwise hangs a bare 2-rank `all_reduce`), `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300` and the flight recorder so a collective mismatch fails inside the 15 min limit with a trace in `<result_dir>/nccl_trace_rank_*`. Every knob is `${VAR:-default}` and can be overridden from the submitting shell. |
| `script.py` | Per-rank entrypoint. Builds the Hydra override list (`experiment=star_easy_mlm debug=smoke trainer_strategy=fsdp` + per-`RotaryTransformerLayer` wrap policy + the probe callback), calls `xlm.commands.lightning_main.main` in-process, and writes `rank_<RANK>.json` with the probe output and any exception the run raised. |
| `probe.py` | `FSDPCoverageProbe` Lightning callback: records trainable vs optimizer numel, the coverage-check outcome, root-unit checksums before/after training, and local vs global vs logged gradient norms. |

## Scenarios (one sbatch each)

| Test | `lightning_module` | `use_orig_params` | Expected |
| --- | --- | --- | --- |
| `test_fsdp_harness_default_config_trains_all_parameters` | `FSDPHarness` | `true` (default) | 5 steps, coverage ok, root unit moved, logged grad norm == all-reduced global norm |
| `test_base_harness_with_orig_params_trains_all_parameters` | `Harness` | `true` | 5 steps, coverage ok, root unit moved (clipping off: `FSDPPrecision` rejects norm clipping without `FSDPHarness`) |
| `test_base_harness_without_orig_params_fails_fast` | `Harness` | `false` | `RuntimeError` from `check_optimizer_covers_trainable_params` naming the root `_flat_param`; training never starts |

## Result-file schema

```json
{
  "rank": 0,
  "world_size": 2,
  "ok": true,
  "error": null,
  "config": {"harness": "xlm.harness.FSDPHarness", "use_orig_params": true, "gradient_clip_val": 1.0},
  "train_error": null,
  "train_error_type": null,
  "probe": {
    "harness_class": "FSDPHarness",
    "is_fsdp": true,
    "use_orig_params": true,
    "trainable_numel": 123456,
    "optimizer_numel": 123456,
    "coverage_ok": true,
    "coverage_error": "",
    "num_root_unit_params": 4,
    "root_checksums_before": {"...": 0.0},
    "root_checksums_after": {"...": 0.0},
    "root_moved": true,
    "global_step": 5,
    "local_grad_norms": [1.0],
    "global_grad_norms": [1.4],
    "logged_total_grad_norms": [1.4]
  }
}
```

`ok=false` means `script.py` itself failed (import error, bad JSON, ...).
A training exception is reported in `train_error` / `train_error_type` with
`ok=true`, so the pytest can assert on *expected* failures.
