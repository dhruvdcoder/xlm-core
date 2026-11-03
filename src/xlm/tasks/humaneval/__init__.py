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
    strip: bool = True,
) -> Dict[str, Any]:
    """Prompt prefix (prompt.strip() + sep) with an empty target, as in dream_code_adaptation training.

    ``strip=False, sep=""`` keeps the raw HumanEval prompt (FlexMDM reference encoding).
    """
    prompt = example.get("prompt") or ""
    if strip:
        prompt = prompt.strip()
    example["prompt_ids"] = tokenizer.encode(prompt, add_special_tokens=False)
    if sep:
        example["prompt_ids"] = example["prompt_ids"] + tokenizer.encode(
            sep, add_special_tokens=False
        )
    example["input_ids"] = []
    return example
