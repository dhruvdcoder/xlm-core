"""pass@k scoring for HumanEval / HumanEval+.

Candidate extraction, program assembly, sandboxed execution and the pass@k
estimator follow FlexMDM's ``evals/humaneval_compare/passk.py``; the
extraction-robust (any-of-4 modes) score follows ``robust_passk.py``.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from xlm.tasks.humaneval.data import (
    ALL_VARIANTS,
    VARIANT_BASE,
    VARIANT_PLUS,
    HumanEvalTask,
    load_humaneval_tasks,
)
from xlm.tasks.humaneval.sanitize import sanitize
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

FENCE = "`" * 3

ALL_MODES = (
    "prompt_tail_sanitize",
    "prompt_cleaned_sanitize",
    "prompt_tail_raw",
    "cleaned_raw",
)
DEFAULT_KS = (1, 2, 4, 8, 16)
VARIANT_TO_DATASET = {VARIANT_BASE: "humaneval", VARIANT_PLUS: "humaneval_plus"}


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def split_tail_cleaned(completion: str) -> Tuple[str, str]:
    cleaned = completion.split(FENCE, 1)[0].strip()
    return completion, cleaned


def _safe_sanitize(source: str, entry_point: str) -> Tuple[str, Optional[str]]:
    try:
        return sanitize(source, entry_point), None
    except Exception:
        return "", traceback.format_exc(limit=1).strip()


def build_candidate_code(
    *, task: HumanEvalTask, completion: str, mode: str
) -> Tuple[str, Optional[str]]:
    """HumanEval candidate for ``mode``; the prompt skeleton is the prefix."""
    if mode not in ALL_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {ALL_MODES}.")
    prefix = task.prompt
    tail, cleaned = split_tail_cleaned(completion)
    if mode == "prompt_tail_sanitize":
        source = (prefix + "\n" + tail) if prefix else tail
        return _safe_sanitize(source, task.entry_point)
    if mode == "prompt_cleaned_sanitize":
        source = (prefix + "\n" + cleaned) if prefix else cleaned
        return _safe_sanitize(source, task.entry_point)
    if mode == "prompt_tail_raw":
        cut = tail.split(FENCE, 1)[0]
        return ((prefix + "\n" + cut) if prefix else cut), None
    if mode == "cleaned_raw":
        return cleaned, None
    raise AssertionError(f"Unhandled mode {mode!r}.")


def build_program(*, task: HumanEvalTask, candidate_code: str, variant: str) -> str:
    return (
        f"{candidate_code}\n\n"
        f"{task.test_for(variant)}\n\n"
        f"check({task.entry_point})\n"
    )


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

_RUNNER_PROLOGUE = (
    "import math\n"
    "import re\n"
    "import sys\n"
    "import time\n"
    "import itertools\n"
    "import functools\n"
    "import collections\n"
    "from typing import *\n"
    "\n"
)


def run_one(
    program: str, *, timeout_s: float, python_bin: Optional[str] = None
) -> Tuple[bool, str, str]:
    """Run ``program`` via ``python -I -`` (stdin avoids argv limits on HE+ tests)."""
    binary = python_bin or sys.executable
    full = _RUNNER_PROLOGUE + program
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    try:
        proc = subprocess.run(
            [binary, "-I", "-"],
            input=full,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout", f"Timed out after {timeout_s}s"
    except Exception as exc:
        return False, "subprocess_error", repr(exc)

    if proc.returncode == 0:
        return True, "passed", ""
    err = (proc.stderr or proc.stdout or "").strip()
    if "SyntaxError" in err:
        kind = "syntax_error"
    elif "AssertionError" in err:
        kind = "assertion_error"
    elif "NameError" in err:
        kind = "name_error"
    elif "ImportError" in err or "ModuleNotFoundError" in err:
        kind = "import_error"
    else:
        kind = "runtime_error"
    return False, kind, err[-1200:]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al., 2021) from ``n`` samples with ``c`` correct."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if n < 1:
        raise ValueError(f"n must be >=1, got {n}")
    if c < 0 or c > n:
        raise ValueError(f"c must be in [0, n]; got n={n}, c={c}")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# ---------------------------------------------------------------------------
# Per-task worker
# ---------------------------------------------------------------------------


def _evaluate_task(
    task: HumanEvalTask,
    samples: List[Tuple[int, str]],
    modes: Sequence[str],
    variants: Sequence[str],
    timeout_s: float,
) -> List[Dict[str, Any]]:
    """Score every (sample, mode, variant) of one task; one row per combination."""
    rows: List[Dict[str, Any]] = []
    for sample_k, completion in samples:
        for mode in modes:
            candidate, sanitize_err = build_candidate_code(
                task=task, completion=completion, mode=mode
            )
            for variant in variants:
                if sanitize_err:
                    passed, err_type, err = False, "sanitize_error", sanitize_err
                else:
                    program = build_program(
                        task=task, candidate_code=candidate, variant=variant
                    )
                    passed, err_type, err = run_one(program, timeout_s=timeout_s)
                rows.append(
                    {
                        "task_id": task.task_id,
                        "sample_k": sample_k,
                        "mode": mode,
                        "variant": variant,
                        "passed": passed,
                        "error_type": err_type,
                        "error": err,
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Completion extraction
# ---------------------------------------------------------------------------


def prediction_completion(pred: Dict[str, Any], task: HumanEvalTask) -> str:
    """Completion text after the prompt.

    Prefers ``generated_text``. Otherwise strips the encoded prompt
    (``prompt.strip() + "\\n"``) from the front of the full decode ``text``.
    """
    if pred.get("generated_text") is not None:
        return str(pred["generated_text"])
    text = str(pred.get("text", ""))
    prompt_text = task.prompt.strip() + "\n"
    if not text.startswith(prompt_text):
        raise ValueError(
            f"{task.task_id}: decoded text does not start with the prompt; "
            "log `generated_text` from the predictor instead."
        )
    return text[len(prompt_text) :]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class HumanEvalPassKEval:
    """Post-hoc evaluator: pass@k on HumanEval (base) and HumanEval+ (plus).

    Expects prediction rows with ``task_id``, ``sample_k`` and either
    ``generated_text`` or ``text``. Reports, per extraction mode and variant,
    ``{humaneval,humaneval_plus}/{mode}/pass@k`` plus the extraction-robust
    ``{humaneval,humaneval_plus}/robust/pass@k`` (a sample passes if any mode
    passes). When ``output_dir`` is set, also writes ``per_sample.jsonl``,
    ``summary.json`` and ``robust_summary.json`` in the reference layout.

    Hydra::

        post_hoc_evaluator:
          _target_: xlm.tasks.humaneval.HumanEvalPassKEval
    """

    def __init__(
        self,
        n_samples: int = 16,
        ks: Sequence[int] = DEFAULT_KS,
        modes: Sequence[str] = ALL_MODES,
        variants: Sequence[str] = ALL_VARIANTS,
        timeout_s: float = 30.0,
        workers: int = 16,
        output_dir: Optional[str] = None,
        model_name: str = "model",
        alg: str = "default",
    ) -> None:
        for mode in modes:
            if mode not in ALL_MODES:
                raise ValueError(f"Unknown mode {mode!r}; expected {ALL_MODES}.")
        for variant in variants:
            if variant not in ALL_VARIANTS:
                raise ValueError(
                    f"Unknown variant {variant!r}; expected {ALL_VARIANTS}."
                )
        self.n_samples = n_samples
        self.ks = tuple(int(k) for k in ks)
        self.modes = tuple(modes)
        self.variants = tuple(variants)
        self.timeout_s = timeout_s
        self.workers = workers
        self.output_dir = output_dir
        self.model_name = model_name
        self.alg = alg

    def _group(
        self, predictions: List[Dict[str, Any]], tasks: Dict[str, HumanEvalTask]
    ) -> Dict[str, List[Tuple[int, str]]]:
        grouped: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for pred in predictions:
            task_id = str(pred["task_id"])
            if task_id not in tasks:
                raise KeyError(f"Unknown HumanEval task_id {task_id!r}")
            completion = prediction_completion(pred, tasks[task_id])
            pred["completion"] = completion
            grouped[task_id].append((int(pred["sample_k"]), completion))
        bad = {
            tid: sorted(k for k, _ in rows)
            for tid, rows in grouped.items()
            if sorted(k for k, _ in rows) != list(range(self.n_samples))
        }
        if bad:
            example = next(iter(bad.items()))
            raise ValueError(
                f"{len(bad)} task(s) do not have exactly sample_k=0..{self.n_samples - 1} "
                f"(e.g. {example[0]}: {example[1]}). Duplicate or missing samples "
                "would bias pass@k."
            )
        return grouped

    def _score(
        self, grouped: Dict[str, List[Tuple[int, str]]], tasks: Dict[str, HumanEvalTask]
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(
            max_workers=self.workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(
                    _evaluate_task,
                    tasks[tid],
                    sorted(samples),
                    self.modes,
                    self.variants,
                    self.timeout_s,
                )
                for tid, samples in grouped.items()
            ]
            for fut in futures:
                rows.extend(fut.result())
        return rows

    def _passk(self, n_correct_by_task: Dict[str, int], n_tasks: int) -> Dict[int, float]:
        return {
            k: sum(
                pass_at_k(self.n_samples, n_correct_by_task.get(tid, 0), k)
                for tid in sorted(n_correct_by_task)
            )
            / n_tasks
            for k in self.ks
            if k <= self.n_samples
        }

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not predictions:
            return predictions, {}
        tasks = {t.task_id: t for t in load_humaneval_tasks()}
        grouped = self._group(predictions, tasks)
        task_ids = sorted(grouped)
        n_tasks = len(task_ids)
        rows = self._score(grouped, tasks)

        # (mode, variant) -> task -> #passed ; variant -> task -> {sample_k passing any mode}
        per_mode: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: {tid: 0 for tid in task_ids}
        )
        any_pass: Dict[str, Dict[str, set]] = defaultdict(
            lambda: {tid: set() for tid in task_ids}
        )
        errors: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        sample_results: Dict[Tuple[str, int], Dict[str, Dict[str, bool]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for r in rows:
            key = (r["mode"], r["variant"])
            _ = per_mode[key]
            if r["passed"]:
                per_mode[key][r["task_id"]] += 1
                any_pass[r["variant"]][r["task_id"]].add(r["sample_k"])
            else:
                errors[key][r["error_type"]] += 1
            sample_results[(r["task_id"], r["sample_k"])][r["mode"]][r["variant"]] = r[
                "passed"
            ]

        metrics: Dict[str, Any] = {"n_tasks": n_tasks, "n_samples": self.n_samples}
        summary: Dict[str, Any] = {}
        robust_summary: Dict[str, Any] = {}
        for variant in self.variants:
            dataset = VARIANT_TO_DATASET[variant]
            per_mode_passk: Dict[str, Dict[int, float]] = {}
            for mode in self.modes:
                counts = per_mode[(mode, variant)]
                passk = self._passk(counts, n_tasks)
                per_mode_passk[mode] = passk
                for k, v in passk.items():
                    metrics[f"{dataset}/{mode}/pass@{k}"] = v
                row: Dict[str, Any] = {
                    "n_units": n_tasks,
                    "n_samples_total": n_tasks * self.n_samples,
                    "n_correct_total": sum(counts.values()),
                }
                row.update({f"pass@{k}": v for k, v in passk.items()})
                row["error_counts"] = dict(errors[(mode, variant)])
                summary[
                    f"humaneval|{self.model_name}|{self.alg}|mode={mode}|variant={variant}"
                ] = row
            robust = self._passk(
                {tid: len(s) for tid, s in any_pass[variant].items()}, n_tasks
            )
            for k, v in robust.items():
                metrics[f"{dataset}/robust/pass@{k}"] = v
            robust_summary.setdefault(f"humaneval|{self.model_name}|{self.alg}", {})[
                variant
            ] = {
                "n_tasks": n_tasks,
                "n_samples": self.n_samples,
                "n_samples_seen": self.n_samples,
                "robust": robust,
                "per_mode": per_mode_passk,
            }

        for pred in predictions:
            pred["humaneval_pass"] = {
                mode: dict(v)
                for mode, v in sample_results[
                    (str(pred["task_id"]), int(pred["sample_k"]))
                ].items()
            }

        for variant in self.variants:
            dataset = VARIANT_TO_DATASET[variant]
            logger.info(
                f"HumanEvalPassKEval {dataset}: "
                + "  ".join(
                    f"pass@{k}={metrics[f'{dataset}/prompt_tail_sanitize/pass@{k}'] * 100:.2f}"
                    for k in self.ks
                    if f"{dataset}/prompt_tail_sanitize/pass@{k}" in metrics
                )
                + " (prompt_tail_sanitize) | robust "
                + "  ".join(
                    f"pass@{k}={metrics[f'{dataset}/robust/pass@{k}'] * 100:.2f}"
                    for k in self.ks
                    if f"{dataset}/robust/pass@{k}" in metrics
                )
            )

        if self.output_dir:
            self._write_outputs(rows, summary, robust_summary)
        return predictions, metrics

    def _write_outputs(
        self,
        rows: List[Dict[str, Any]],
        summary: Dict[str, Any],
        robust_summary: Dict[str, Any],
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "per_sample.jsonl"), "w") as f:
            for r in rows:
                f.write(
                    json.dumps(
                        {
                            "task_id": r["task_id"],
                            "gen_dataset": "humaneval",
                            "model": self.model_name,
                            "alg": self.alg,
                            "mode": r["mode"],
                            "variant": r["variant"],
                            "sample_k": r["sample_k"],
                            "passed": bool(r["passed"]),
                        }
                    )
                    + "\n"
                )
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(self.output_dir, "robust_summary.json"), "w") as f:
            json.dump(robust_summary, f, indent=2)
        logger.info(f"HumanEvalPassKEval: wrote results under {self.output_dir}")
