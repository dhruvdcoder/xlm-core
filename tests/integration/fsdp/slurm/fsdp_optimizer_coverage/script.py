"""Per-rank entrypoint for the 2-GPU FSDP optimizer-coverage SLURM test.

Runs the *real* xlm training path in-process: Hydra composition with the
tests/configs debug=smoke overlay, trainer_strategy=fsdp, a per-block
auto_wrap_policy on the MLM backbone layer, Lightning's FSDPStrategy wrapping
the Harness, and Harness.configure_optimizers running after the wrap. The
FSDPCoverageProbe callback records what happened; this script serialises the
probe output plus any exception the run raised to rank_<RANK>.json.

A run that raises is *not* a script failure: the parent pytest decides
whether the exception was expected (for example the coverage RuntimeError
when use_orig_params=false is combined with a model whose parameter groups
only look at harness.model). The result file therefore always has ok=True
unless this script itself could not do its job.

Config schema (JSON, via --config-json)::

    {
      "harness": "xlm.harness.FSDPHarness",   # or xlm.harness.Harness
      "use_orig_params": true,
      "gradient_clip_val": 1.0,               # null disables clipping
      "extra_overrides": ["..."]              # optional Hydra overrides
    }

Environment: RANK / WORLD_SIZE / LOCAL_RANK are read by Lightning's
SLURMEnvironment from SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID; script.sh
also exports MASTER_ADDR / MASTER_PORT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
TESTS_CONFIG_DIR = REPO_ROOT / "tests" / "configs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--config-json", type=str, default="{}")
    return parser.parse_args()


def _overrides(config: Dict[str, Any], result_dir: Path) -> List[str]:
    harness = str(config.get("harness", "xlm.harness.FSDPHarness"))
    use_orig_params = bool(config.get("use_orig_params", True))
    clip = config.get("gradient_clip_val", 1.0)
    overrides = [
        "--config-dir",
        str(TESTS_CONFIG_DIR),
        "job_type=train",
        "job_name=fsdp_optimizer_coverage",
        "experiment=star_easy_mlm",
        "debug=smoke",
        "trainer_strategy=fsdp",
        "trainer.accelerator=cuda",
        "trainer.devices=2",
        "trainer.num_nodes=1",
        # debug=smoke sets global_batch_size=1; two ranks need 2.
        "global_batch_size=2",
        "per_device_batch_size=1",
        f"lightning_module._target_={harness}",
        f"trainer.strategy.use_orig_params={'true' if use_orig_params else 'false'}",
        "++trainer.strategy.auto_wrap_policy._target_="
        "xlm.utils.fsdp_grouping.make_layer_wrap_policy",
        "++trainer.strategy.auto_wrap_policy._args_="
        "[xlm.modules.rotary_transformer.RotaryTransformerLayer]",
        "++trainer.gradient_clip_algorithm=norm",
        f"trainer.gradient_clip_val={'null' if clip is None else clip}",
        # EMA keeps a CPU shadow of parameters; irrelevant here and not
        # FSDP-aware, so drop it from the debug_v2 callback set.
        "~callbacks.ema",
        "++callbacks.fsdp_probe._target_="
        "tests.integration.fsdp.slurm.fsdp_optimizer_coverage.probe.FSDPCoverageProbe",
        f"paths.log_dir={result_dir / 'logs'}",
    ]
    overrides.extend(str(o) for o in config.get("extra_overrides", []))
    return overrides


def main() -> None:
    args = _parse_args()
    config = json.loads(args.config_json)
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    args.result_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "rank": rank,
        "world_size": world_size,
        "ok": True,
        "error": None,
        "config": config,
        "train_error": None,
        "train_error_type": None,
    }
    try:
        sys.path.insert(0, str(REPO_ROOT))
        os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT))
        os.environ["HYDRA_FULL_ERROR"] = "1"
        from tests.integration.fsdp.slurm.fsdp_optimizer_coverage import probe

        sys.argv = ["xlm", *_overrides(config, args.result_dir)]
        from xlm.commands.lightning_main import main as xlm_main

        try:
            xlm_main()
        except SystemExit as exc:  # hydra may sys.exit on error
            if exc.code not in (0, None):
                payload["train_error"] = f"SystemExit({exc.code})"
                payload["train_error_type"] = "SystemExit"
        except Exception as exc:  # noqa: BLE001 - recorded for pytest
            payload["train_error"] = traceback.format_exc()[-4000:]
            payload["train_error_type"] = type(exc).__name__
            payload["train_error_message"] = str(exc)[:1000]
        payload["probe"] = dict(probe.RESULT)
    except Exception:  # noqa: BLE001 - genuine script failure
        payload["ok"] = False
        payload["error"] = traceback.format_exc()

    with (args.result_dir / f"rank_{rank}.json").open("w") as f:
        json.dump(payload, f, indent=2, default=str)


if __name__ == "__main__":
    main()
