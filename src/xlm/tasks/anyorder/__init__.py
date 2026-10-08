
from typing import Any, Dict, Optional

_anyorder_debug_kept_prompt: Optional[str] = None


def reset_anyorder_debug_first_example_filter_fn() -> None:
    """Reset :func:`anyorder_debug_first_example_filter_fn` state (for tests)."""
    global _anyorder_debug_kept_prompt
    _anyorder_debug_kept_prompt = None


def anyorder_debug_first_example_filter_fn(example: Dict[str, Any]) -> bool:
    """Keep only the first any-order row when building debug manual caches.

    Used with ``filter_suffix: debug_one`` in the any-order debug dataset configs.
    The kept row is remembered by its prompt, so later passes over the same split
    (e.g. the ``debug_one_pred`` cache) select the same row.
    Run ``prepare_data`` with ``num_dataset_workers=1`` so ``Dataset.filter`` is
    single-process; multiprocessing can drop or duplicate rows.
    """
    global _anyorder_debug_kept_prompt
    prompt = example.get("prompt")
    if _anyorder_debug_kept_prompt is None:
        _anyorder_debug_kept_prompt = prompt
        return True
    return prompt == _anyorder_debug_kept_prompt


def anyorder_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: Any,
    *,
    sep: str = "\n",
) -> Dict[str, Any]:
    """Tokenize any-order rows into prompt/target token id lists.

    Args:
        example: HF row with ``prompt`` and ``answer`` fields.
        tokenizer: Hugging Face tokenizer (``encode``, no special tokens).
        sep: String between prompt and answer.
    """
    question = (example.get("prompt") or "").strip()
    response = (example.get("answer") or "").strip()
    sep_ids = tokenizer.encode(sep, add_special_tokens=False)
    p_ids = tokenizer.encode(question, add_special_tokens=False)
    r_ids = tokenizer.encode(response, add_special_tokens=False)
    example["prompt_token_ids"] = p_ids + sep_ids
    example["input_token_ids"] = r_ids
    return example


def anyorder_pred_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: Any,
    *,
    sep: str = "\n",
) -> Dict[str, Any]:
    """Prediction rows: prompt prefix, empty target, reference kept in ``answer``."""
    question = (example.get("prompt") or "").strip()
    response = (example.get("answer") or "").strip()
    sep_ids = tokenizer.encode(sep, add_special_tokens=False)
    p_ids = tokenizer.encode(question, add_special_tokens=False)
    example["prompt_token_ids"] = p_ids + sep_ids
    example["input_token_ids"] = []
    example["answer"] = response
    return example
