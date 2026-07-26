"""LLaDA-specific data utilities (batch printing, OpenMath SFT preprocess)."""

from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizerBase

from xlm.datamodule import Tokenizer
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def print_batch_llada(
    batch: Dict[str, Any],
    split: str,
    tokenizer: Tokenizer,
    dataloader_name: str = "",
) -> None:
    """Debug helper that logs the first example of a LLaDA / MLM prediction batch."""
    ids = batch["input_ids"][0]
    mask = batch["attention_mask"][0]
    text = tokenizer.decode(ids[mask.bool()], skip_special_tokens=False)
    logger.info(
        f"[LLaDA] split={split}  dl_name={dataloader_name}  "
        f"shape={tuple(batch['input_ids'].shape)}  "
        f"prompt_preview={text}"
    )


def openmath_raw_filter_fn(example: Dict[str, Any]) -> bool:
    """Keep rows with non-empty problem + solution before tokenization."""
    problem = str(example.get("problem") or "").strip()
    solution = str(
        example.get("generated_solution") or example.get("solution") or ""
    ).strip()
    return bool(problem) and bool(solution)


def _tokenize_prompt_answer(
    prompt: str,
    answer: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    sep: str = "\n",
    max_length: int = 2048,
) -> Dict[str, List[int]]:
    """Shared prompt/answer tokenization with answer-first truncation."""
    sep_ids = tokenizer.encode(sep, add_special_tokens=False)
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(answer, add_special_tokens=False)
    prompt_token_ids = p_ids + sep_ids
    truncated = False
    if len(prompt_token_ids) >= max_length:
        prompt_token_ids = prompt_token_ids[: max(1, max_length - 1)]
        a_ids = a_ids[:1] if a_ids else [tokenizer.eos_token_id or 0]
        truncated = True
    else:
        budget = max_length - len(prompt_token_ids)
        if len(a_ids) > budget:
            truncated = True
        a_ids = a_ids[:budget]
    return {
        "prompt_token_ids": prompt_token_ids,
        "input_token_ids": a_ids,
        "truncated": truncated,
    }


def openmath_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    sep: str = "\n",
    max_length: int = 2048,
) -> Dict[str, Any]:
    """Tokenize OpenMathInstruct-2 rows into prompt/answer id lists.

    Fields: ``problem`` → prompt, ``generated_solution`` (fallback ``solution``)
    → answer. Truncates the answer to fit ``max_length`` (prompt kept intact
    when possible; if the prompt alone exceeds the budget, both sides are
    truncated).
    """
    problem = (example.get("problem") or "").strip()
    solution = (
        example.get("generated_solution") or example.get("solution") or ""
    ).strip()
    tok = _tokenize_prompt_answer(
        problem, solution, tokenizer, sep=sep, max_length=max_length
    )
    example["prompt_token_ids"] = tok["prompt_token_ids"]
    example["input_token_ids"] = tok["input_token_ids"]
    return example


def mix_raw_filter_fn(example: Dict[str, Any]) -> bool:
    """Keep rows with non-empty plain prompt + answer (no ChatML)."""
    prompt = str(example.get("prompt") or "").strip()
    answer = str(example.get("answer") or "").strip()
    return bool(prompt) and bool(answer)


def mix_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    sep: str = "\n",
    max_length: int = 2048,
) -> Dict[str, Any]:
    """Tokenize Fast-dLLM c40m60 plain ``prompt`` / ``answer`` rows for LLaDA."""
    prompt = (example.get("prompt") or "").strip()
    answer = (example.get("answer") or "").strip()
    tok = _tokenize_prompt_answer(
        prompt, answer, tokenizer, sep=sep, max_length=max_length
    )
    example["prompt_token_ids"] = tok["prompt_token_ids"]
    example["input_token_ids"] = tok["input_token_ids"]
    return example


def print_batch_openmath_llada(
    batch: Dict[str, Any],
    split: str,
    tokenizer: Tokenizer,
    dataloader_name: str = "",
) -> None:
    """Log a short preview of an OpenMath SFT training batch."""
    ids = batch["input_ids"][0]
    attn = batch["attention_mask"][0].bool()
    masked = (ids == tokenizer.mask_token_id) & attn
    text = tokenizer.decode(ids[attn], skip_special_tokens=False)
    logger.info(
        f"[LLaDA-OpenMath] split={split} dl={dataloader_name} "
        f"shape={tuple(batch['input_ids'].shape)} "
        f"n_masked={int(masked.sum())} preview={text[:240]!r}"
    )


def print_batch_mix_llada(
    batch: Dict[str, Any],
    split: str,
    tokenizer: Tokenizer,
    dataloader_name: str = "",
) -> None:
    """Log a short preview of an OpenCode+OpenMath mix SFT batch."""
    ids = batch["input_ids"][0]
    attn = batch["attention_mask"][0].bool()
    masked = (ids == tokenizer.mask_token_id) & attn
    text = tokenizer.decode(ids[attn], skip_special_tokens=False)
    logger.info(
        f"[LLaDA-c40m60] split={split} dl={dataloader_name} "
        f"shape={tuple(batch['input_ids'].shape)} "
        f"n_masked={int(masked.sum())} preview={text[:240]!r}"
    )
