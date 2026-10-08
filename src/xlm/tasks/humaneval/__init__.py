"""HumanEval / HumanEval+ code generation eval (pass@k).

Data and scoring follow FlexMDM's ``evals/humaneval_compare``: HumanEval
prompts with base and HumanEval+ tests (:mod:`data`), four completion
extraction modes, sandboxed execution and unbiased pass@k (:mod:`eval`).
"""

from typing import Any, Dict

from xlm.tasks.humaneval.data import (
    HumanEvalDatasetManager,
    HumanEvalTask,
    load_humaneval_tasks,
)
from xlm.tasks.humaneval.eval import HumanEvalPassKEval, pass_at_k

__all__ = [
    "HumanEvalDatasetManager",
    "HumanEvalPassKEval",
    "HumanEvalTask",
    "humaneval_pred_preprocess_fn",
    "load_humaneval_tasks",
    "pass_at_k",
]


def humaneval_pred_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: Any,
    *,
    sep: str = "\n",
) -> Dict[str, Any]:
    """Prompt prefix (``prompt.strip() + sep``) with an empty target, as in any-order training."""
    prompt = (example.get("prompt") or "").strip()
    example["prompt_ids"] = tokenizer.encode(
        prompt, add_special_tokens=False
    ) + tokenizer.encode(sep, add_special_tokens=False)
    example["input_ids"] = []
    return example
