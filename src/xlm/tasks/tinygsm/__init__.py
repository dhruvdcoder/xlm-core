"""Preprocessing for TinyGSM/TinyGSM (math word problems + Python solutions).

Field layout and train/val split semantics follow PUMA's tiny_gsm.py:
https://github.com/JaeyeonKim01/PUMA/blob/main/data/tiny_gsm.py

Each example is split into a prefix (question + separator) and a suffix (code)
for seq2seq MDM training via ``prompt_token_ids`` / ``input_token_ids`` and an
on-the-fly processor that maps to ``prompt_ids`` / ``input_ids``. Wire a
seq2seq collator (e.g. ``MLMSeq2SeqTrainCollator``) in the model experiment.

Memmap pretokenization (``pretokenize_tinygsm``, ``labels.bin``,
``prompt_mask.bin``, ``TinyGSMDataset``) is not supported and will not be added.
Data flows only through ``DatasetManager`` + ``prepare_data`` + iterable shards.

Padding/truncation to a fixed block size is handled by the collator, not here.
PUMA pads with EOS; xlm collators use ``pad_token_id`` unless the experiment
sets ``loss_on_padding`` or pad=eos on the tokenizer.
"""

from typing import Any, Dict

from transformers import PreTrainedTokenizerBase


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
