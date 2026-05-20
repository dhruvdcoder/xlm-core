"""Preprocessing for TinyGSM/TinyGSM (math word problems + Python solutions).

Field layout and train/val split semantics follow PUMA's tiny_gsm.py:
https://github.com/JaeyeonKim01/PUMA/blob/main/data/tiny_gsm.py

Each example is split into a prefix (question + separator) and a suffix (code)
for seq2seq MDM training via ``prompt_token_ids`` / ``input_token_ids`` and an
on-the-fly processor that maps to ``prompt_ids`` / ``input_ids``. Wire a
seq2seq collator (e.g. ``MLMSeq2SeqTrainCollator``) in the model experiment.

GSM8K test evaluation (code execution scoring) lives in :mod:`gsm8k` — see
``gsm8k_preprocess_fn`` and ``Gsm8kCodeEval`` (PUMA gsm8k_eval.py).

Memmap pretokenization (``pretokenize_tinygsm``, ``labels.bin``,
``prompt_mask.bin``, ``TinyGSMDataset``) is not supported and will not be added.
Data flows only through ``DatasetManager`` + ``prepare_data`` + iterable shards.

Padding/truncation to a fixed block size is handled by the collator, not here.
PUMA pads with EOS; xlm collators use ``pad_token_id`` unless the experiment
sets ``loss_on_padding`` or pad=eos on the tokenizer.
"""

from typing import Any, Dict

from transformers import PreTrainedTokenizerBase

from xlm.tasks.tinygsm.gsm8k import (
    Gsm8kCodeEval,
    evaluate_samples,
    extract_gsm8k_final_answer,
    gold_answer_from_tinygsm_code,
    gsm8k_preprocess_fn,
    tinygsm_pred_preprocess_fn,
)

__all__ = [
    "Gsm8kCodeEval",
    "evaluate_samples",
    "extract_gsm8k_final_answer",
    "gold_answer_from_tinygsm_code",
    "gsm8k_preprocess_fn",
    "reset_tinygsm_debug_first_example_filter_fn",
    "tinygsm_debug_first_example_filter_fn",
    "tinygsm_pred_preprocess_fn",
    "tinygsm_preprocess_fn",
]

_tinygsm_debug_first_example_kept = False


def reset_tinygsm_debug_first_example_filter_fn() -> None:
    """Reset :func:`tinygsm_debug_first_example_filter_fn` state (for tests)."""
    global _tinygsm_debug_first_example_kept
    _tinygsm_debug_first_example_kept = False


def tinygsm_debug_first_example_filter_fn(example: Dict[str, Any]) -> bool:
    """Keep only the first TinyGSM row when building debug manual caches.

    Used with ``filter_suffix: debug_one`` in flexmdm debug dataset configs.
    Run ``prepare_data`` with ``num_dataset_workers=1`` so ``Dataset.filter`` is
    single-process; multiprocessing can drop or duplicate rows.
    """
    global _tinygsm_debug_first_example_kept
    if _tinygsm_debug_first_example_kept:
        return False
    _tinygsm_debug_first_example_kept = True
    return True


def tinygsm_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    sep: str = "\n",
) -> Dict[str, Any]:
    """Tokenize TinyGSM rows into prefix/suffix token id lists.

    Args:
        example: HF row with ``question`` and ``code`` fields.
        tokenizer: Hugging Face tokenizer (``encode``, no special tokens).
        sep: String between question and code (PUMA default: newline).
    """
    question = (example.get("question") or "").strip()
    code = (example.get("code") or "").strip()
    sep_ids = tokenizer.encode(sep, add_special_tokens=False)
    p_ids = tokenizer.encode(question, add_special_tokens=False)
    a_ids = tokenizer.encode(code, add_special_tokens=False)
    example["prompt_token_ids"] = p_ids + sep_ids
    example["input_token_ids"] = a_ids
    return example
