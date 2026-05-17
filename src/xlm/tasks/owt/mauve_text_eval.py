"""MAUVE post-hoc text evaluation for xlm-core ``Harness`` / ``LogPredictions``.

Computes `MAUVE <https://arxiv.org/abs/2102.01454>`_ between human/reference
strings and model generations using `mauve-text`_ (``import mauve``).

**Human / reference text** can come from:

1. Each prediction row (``truth``, ``reference``, …, or ``reference_field``),
   including decoded ``target_ids`` when the tokenizer is passed; or
2. ``human_text_source: hf_streaming`` — stream strings from a HuggingFace
   dataset split (default: OWT validation), same idea as the standalone Proseco
   eval (human side from the val loader, not the JSONL).

Example Hydra defaults::

    defaults:
      - your_experiment
      - /post_hoc_evaluator: mauve_text

Or instantiate explicitly::

    post_hoc_evaluator:
      _target_: xlm.tasks.owt.mauve_text_eval.MauveTextEval

Install::

    pip install "xlm-core[mauve]"

(or ``pip install mauve-text``).

.. _mauve-text: https://pypi.org/project/mauve-text/
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Convention from the MAUVE README: ``p_text`` = human, ``q_text`` = machine.
_REFERENCE_KEYS: Sequence[str] = (
    "truth",
    "reference",
    "reference_text",
    "target_text",
    "ground_truth",
    "ground_truth_middle",
    "answer",
    "gold",
)


def _decode_target_ids(
    value: Any, tokenizer: Any
) -> Optional[str]:
    if tokenizer is None or value is None:
        return None
    try:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, int):
                ids = value
            elif isinstance(first, (list, tuple)) and first:
                ids = first
            else:
                return None
            if not ids:
                return None
            return tokenizer.decode(ids, skip_special_tokens=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to decode target_ids for MAUVE: %s", exc)
    return None


def _reference_for_pred(
    pred: Dict[str, Any],
    tokenizer: Any,
    reference_field: Optional[str],
) -> str:
    if reference_field is not None:
        raw = pred.get(reference_field)
        if raw is not None:
            if isinstance(raw, str):
                return raw.strip()
            if reference_field == "target_ids":
                dec = _decode_target_ids(raw, tokenizer)
                if dec is not None:
                    return dec.strip()
        return ""

    for key in _REFERENCE_KEYS:
        if key not in pred:
            continue
        val = pred[key]
        if isinstance(val, str) and val.strip():
            return val.strip()

    dec = _decode_target_ids(pred.get("target_ids"), tokenizer)
    if dec:
        return dec.strip()
    return ""


def _stream_hf_texts(
    num_texts: int,
    hf_dataset_path: str,
    split: str,
    text_column: str,
    seed: int,
    shuffle_buffer_size: int,
    min_chars: int,
) -> List[str]:
    """Draw ``num_texts`` non-empty strings from a HF dataset (streaming)."""
    from datasets import load_dataset

    if num_texts <= 0:
        return []

    try:
        ds = load_dataset(
            hf_dataset_path,
            split=split,
            streaming=True,
        )
    except Exception as exc:
        logger.error(
            "MauveTextEval: could not load dataset %s split=%r: %s",
            hf_dataset_path,
            split,
            exc,
        )
        return []

    buf = max(int(shuffle_buffer_size), num_texts * 20, 10_000)
    ds = ds.shuffle(seed=seed, buffer_size=buf)
    out: List[str] = []
    for row in ds:
        if len(out) >= num_texts:
            break
        text = row.get(text_column)
        if not text or not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < min_chars:
            continue
        out.append(text)
    if len(out) < num_texts:
        logger.warning(
            "MauveTextEval: hf_streaming only collected %d/%d texts "
            "(dataset exhausted or filters too strict).",
            len(out),
            num_texts,
        )
    return out


class MauveTextEval:
    """Post-hoc evaluator: MAUVE between references and model ``text``.

    Args:
        reference_field: Batch / prediction key for human text. If ``None``,
            the first non-empty among
            ``truth``, ``reference``, ``target_text``, ``ground_truth_middle``,
            etc., or decoded ``target_ids`` when ``tokenizer`` is passed.
        generated_field: Key for model output (default ``text``).
        featurize_model_name: HF model name for MAUVE features (see ``mauve-text``).
        device_id: GPU id for featurization, or ``-1`` for CPU.
        max_text_length: Max tokens per string for the featurizer.
        batch_size: Featurization batch size.
        verbose: Forwarded to ``mauve.compute_mauve``.
        num_buckets: Histogram size (``"auto"`` or int).
        seed: RNG seed for k-means.
        swap_p_q: If ``True``, treat generations as ``p_text`` and references
            as ``q_text`` (library default is human ``p``, machine ``q``).
        human_text_source: If ``\"hf_streaming\"``, build ``p_text`` from a HF
            dataset split instead of per-row references (Proseco-style). If
            ``None``, use ``truth`` / ``reference_field`` / etc. on each row.
        hf_dataset_path: Dataset id for streaming (OWT default).
        hf_split: Split name, e.g. ``validation``.
        hf_text_column: Column with raw text.
        hf_shuffle_seed: Seed for streaming shuffle.
        hf_shuffle_buffer_size: Shuffle buffer for streaming.
        hf_min_chars: Skip shorter snippets.
    """

    def __init__(
        self,
        reference_field: Optional[str] = None,
        generated_field: str = "text",
        featurize_model_name: str = "gpt2-large",
        device_id: int = 0,
        max_text_length: int = 256,
        batch_size: int = 8,
        verbose: bool = False,
        num_buckets: Any = "auto",
        seed: int = 25,
        swap_p_q: bool = False,
        human_text_source: Optional[str] = None,
        hf_dataset_path: str = "dhruveshpatel/owt-gpt2-1024-split",
        hf_split: str = "validation",
        hf_text_column: str = "text",
        hf_shuffle_seed: int = 42,
        hf_shuffle_buffer_size: int = 10000,
        hf_min_chars: int = 8,
    ) -> None:
        self.reference_field = reference_field
        self.generated_field = generated_field
        self.featurize_model_name = featurize_model_name
        self.device_id = device_id
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.verbose = verbose
        self.num_buckets = num_buckets
        self.seed = seed
        self.swap_p_q = swap_p_q
        self.human_text_source = human_text_source
        self.hf_dataset_path = hf_dataset_path
        self.hf_split = hf_split
        self.hf_text_column = hf_text_column
        self.hf_shuffle_seed = hf_shuffle_seed
        self.hf_shuffle_buffer_size = hf_shuffle_buffer_size
        self.hf_min_chars = hf_min_chars

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        del kwargs  # Harness forwards ``dataloader_name``; unused here.

        if not predictions:
            return predictions, {}

        try:
            import mauve
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MAUVE post-hoc eval requires the mauve-text package. "
                'Install with: pip install "xlm-core[mauve]" or pip install mauve-text'
            ) from exc

        use_hf = (self.human_text_source or "").lower() == "hf_streaming"

        p_text: List[str] = []
        q_text: List[str] = []
        kept_indices: List[int] = []

        if use_hf:
            indexed_gens: List[Tuple[int, str]] = []
            for i, pred in enumerate(predictions):
                gen = pred.get(self.generated_field, "")
                if not isinstance(gen, str):
                    gen = str(gen) if gen is not None else ""
                gen = gen.strip()
                if not gen:
                    pred["mauve_included"] = False
                    pred["mauve_skip_reason"] = "empty_generation"
                    continue
                indexed_gens.append((i, gen))

            n_gen = len(indexed_gens)
            if n_gen < 2:
                logger.warning(
                    "MauveTextEval (hf_streaming): need >=2 generations; got %d.",
                    n_gen,
                )
                for i, _ in indexed_gens:
                    predictions[i]["mauve_included"] = False
                    predictions[i]["mauve_skip_reason"] = "too_few_generations"
                return predictions, {
                    "mauve": float("nan"),
                    "mauve_num_pairs": n_gen,
                }

            p_text = _stream_hf_texts(
                n_gen,
                self.hf_dataset_path,
                self.hf_split,
                self.hf_text_column,
                self.hf_shuffle_seed,
                self.hf_shuffle_buffer_size,
                self.hf_min_chars,
            )
            if len(p_text) < n_gen:
                logger.warning(
                    "MauveTextEval (hf_streaming): not enough HF texts (%d < %d); "
                    "skipping MAUVE.",
                    len(p_text),
                    n_gen,
                )
                for i, _ in indexed_gens:
                    predictions[i]["mauve_included"] = False
                    predictions[i]["mauve_skip_reason"] = "hf_stream_too_short"
                return predictions, {
                    "mauve": float("nan"),
                    "mauve_num_pairs": min(len(p_text), n_gen),
                }

            q_text = [g for _, g in indexed_gens]
            p_text = p_text[:n_gen]
            for k, (i, _) in enumerate(indexed_gens):
                pred = predictions[i]
                pred["mauve_included"] = True
                pred["mauve_reference_source"] = "hf_streaming"
                pred["mauve_reference_split"] = self.hf_split
                pred["mauve_reference_excerpt"] = p_text[k][:200]
                kept_indices.append(i)
            if self.swap_p_q:
                p_text, q_text = q_text, p_text
        else:
            for i, pred in enumerate(predictions):
                gen = pred.get(self.generated_field, "")
                if not isinstance(gen, str):
                    gen = str(gen) if gen is not None else ""
                gen = gen.strip()
                ref = _reference_for_pred(pred, tokenizer, self.reference_field)
                if not gen or not ref:
                    pred["mauve_included"] = False
                    pred["mauve_skip_reason"] = (
                        "empty_generation" if not gen else "empty_reference"
                    )
                    continue
                pred["mauve_included"] = True
                pred["mauve_reference_excerpt"] = ref[:200]
                kept_indices.append(i)
                if self.swap_p_q:
                    p_text.append(gen)
                    q_text.append(ref)
                else:
                    p_text.append(ref)
                    q_text.append(gen)

        n = len(p_text)
        if n < 2:
            logger.warning(
                "MauveTextEval: need at least 2 paired non-empty strings; "
                "got %d. Skipping MAUVE.",
                n,
            )
            return predictions, {"mauve": float("nan"), "mauve_num_pairs": n}

        if n < 500:
            logger.warning(
                "MauveTextEval: MAUVE is unreliable with few samples "
                "(got %d; paper uses ~5000 per side).",
                n,
            )

        out = mauve.compute_mauve(
            p_text=p_text,
            q_text=q_text,
            featurize_model_name=self.featurize_model_name,
            device_id=self.device_id,
            max_text_length=self.max_text_length,
            batch_size=self.batch_size,
            verbose=self.verbose,
            num_buckets=self.num_buckets,
            seed=self.seed,
        )

        mauve_score = float(out.mauve)
        fi = float(out.frontier_integral)
        for i in kept_indices:
            predictions[i]["mauve_score"] = mauve_score

        # Only numeric scalars here — ``Harness.compute_post_hoc_metrics`` logs
        # every key via ``self.log`` (strings are not allowed).
        aggregated: Dict[str, Any] = {
            "mauve": mauve_score,
            "mauve_frontier_integral": fi,
            "mauve_num_pairs": n,
        }
        if hasattr(out, "mauve_star"):
            aggregated["mauve_star"] = float(out.mauve_star)
        if hasattr(out, "frontier_integral_star"):
            aggregated["mauve_frontier_integral_star"] = float(
                out.frontier_integral_star
            )

        if kept_indices:
            meta: Dict[str, Any] = {
                "mauve_featurize_model_name": self.featurize_model_name,
            }
            if use_hf:
                meta["mauve_human_text_source"] = "hf_streaming"
                meta["mauve_human_dataset"] = self.hf_dataset_path
                meta["mauve_human_split"] = self.hf_split
            predictions[kept_indices[0]]["mauve_eval_meta"] = meta

        return predictions, aggregated
