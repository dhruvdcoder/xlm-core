"""HumanEval / HumanEval+ task loading.

Ports the task loader of FlexMDM's ``evals/humaneval_compare/common.py``:
164 problems from ``openai/openai_humaneval`` (test split) joined on
``task_id`` with ``evalplus/humanevalplus`` so every task carries both the
base (``test_base``) and plus (``test_plus``) test code. HumanEval and
HumanEval+ share prompts, so one set of generations is scored against both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import datasets

from xlm.datamodule import EvalDatasetManager
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

VARIANT_BASE = "base"
VARIANT_PLUS = "plus"
ALL_VARIANTS = (VARIANT_BASE, VARIANT_PLUS)


@dataclass(frozen=True)
class HumanEvalTask:
    index: int
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test_base: str
    test_plus: str

    def test_for(self, variant: str) -> str:
        if variant == VARIANT_BASE:
            return self.test_base
        if variant == VARIANT_PLUS:
            return self.test_plus
        raise ValueError(f"Unknown variant {variant!r}.")


def load_humaneval_tasks() -> List[HumanEvalTask]:
    """Load all 164 HumanEval problems with both base and plus tests."""
    he = datasets.load_dataset("openai/openai_humaneval", split="test")
    hep = datasets.load_dataset("evalplus/humanevalplus", split="test")
    plus_by_id = {str(row["task_id"]): row for row in hep}
    tasks: List[HumanEvalTask] = []
    for idx, row in enumerate(he):
        task_id = str(row["task_id"])
        plus_row = plus_by_id.get(task_id, {})
        tasks.append(
            HumanEvalTask(
                index=idx,
                task_id=task_id,
                prompt=str(row["prompt"]),
                entry_point=str(row["entry_point"]),
                canonical_solution=str(row.get("canonical_solution", "")),
                test_base=str(row["test"]),
                test_plus=str(plus_row.get("test", "")),
            )
        )
    return tasks


def build_sample_rows(
    tasks: Sequence[HumanEvalTask], n_samples: int
) -> List[Dict[str, Any]]:
    """One row per (task, sample_k), task-major like the reference work list."""
    return [
        {"task_id": t.task_id, "sample_k": k, "prompt": t.prompt}
        for t in tasks
        for k in range(n_samples)
    ]


class HumanEvalDatasetManager(EvalDatasetManager):
    """Eval manager that expands each HumanEval task into ``n_samples`` rows.

    ``full_name`` is kept for naming only; rows come from
    :func:`load_humaneval_tasks` so the prompts match the reference loader exactly.
    """

    def __init__(
        self,
        *args: Any,
        n_samples: int = 16,
        limit: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples
        self.limit = limit

    def _download(self, num_proc: Optional[int] = None) -> datasets.Dataset:
        tasks = load_humaneval_tasks()
        if self.limit is not None:
            tasks = tasks[: self.limit]
        logger.info(
            f"HumanEvalDatasetManager: {len(tasks)} tasks x {self.n_samples} samples"
        )
        return datasets.Dataset.from_list(build_sample_rows(tasks, self.n_samples))
