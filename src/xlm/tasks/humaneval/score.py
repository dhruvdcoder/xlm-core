"""Score HumanEval / HumanEval+ pass@k from prediction JSONL files.

Use this when generation was sharded across jobs
(``HumanEvalDatasetManager(shard_index=..., num_shards=...)``) and the in-job
post-hoc evaluator was disabled; it concatenates the per-shard predictions and
runs :class:`HumanEvalPassKEval` once over the full sample set.

    python -m xlm.tasks.humaneval.score \\
        --predictions logs/<job>_shard*/runs/*/predictions/test/humaneval_prediction/*.jsonl \\
        --n-samples 16 --output-dir /path/to/humaneval_passk
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from typing import Any, Dict, List

from xlm.tasks.humaneval.eval import (
    ALL_MODES,
    DEFAULT_KS,
    HumanEvalPassKEval,
)
from xlm.tasks.humaneval.data import ALL_VARIANTS

logger = logging.getLogger(__name__)


def read_predictions(patterns: List[str]) -> List[Dict[str, Any]]:
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No prediction files match {pattern!r}")
        paths.extend(matches)
    predictions: List[Dict[str, Any]] = []
    for path in paths:
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        logger.info(f"{path}: {len(rows)} predictions")
        predictions.extend(rows)
    return predictions


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Prediction JSONL files or glob patterns (all shards).",
    )
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--modes", nargs="+", default=list(ALL_MODES))
    p.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS))
    p.add_argument("--timeout-s", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="model")
    p.add_argument("--alg", default="default")
    p.add_argument(
        "--expect-tasks",
        type=int,
        default=None,
        help="Fail unless exactly this many distinct tasks are present (e.g. 164).",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    predictions = read_predictions(args.predictions)
    task_ids = {str(r["task_id"]) for r in predictions}
    logger.info(f"{len(predictions)} predictions over {len(task_ids)} tasks")
    if args.expect_tasks is not None and len(task_ids) != args.expect_tasks:
        raise ValueError(
            f"Expected {args.expect_tasks} tasks, found {len(task_ids)}; "
            "a shard is missing or incomplete."
        )

    evaluator = HumanEvalPassKEval(
        n_samples=args.n_samples,
        ks=args.ks,
        modes=args.modes,
        variants=args.variants,
        timeout_s=args.timeout_s,
        workers=args.workers,
        output_dir=args.output_dir,
        model_name=args.model_name,
        alg=args.alg,
    )
    _, metrics = evaluator.eval(predictions)
    for name in sorted(metrics):
        value = metrics[name]
        if isinstance(value, float):
            logger.info(f"{name}: {value * 100:.2f}")
    return metrics


if __name__ == "__main__":
    main()
