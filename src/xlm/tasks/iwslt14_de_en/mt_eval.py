"""IWSLT14 MT post-hoc evaluation (SacreBLEU + chrF++ / tokenized BLEU).

Scores logged prediction rows after an epoch. Hypothesis strings should be
suffix-only (``generated_text`` from predictor ``to_dict``); full-sequence
``text`` is only a fallback.

Modes (prep plan §8):

* ``sacrebleu_detok`` (primary): Moses-detokenize hyps, refs from ``target_raw``,
  signed SacreBLEU (``tok.13a``) + chrF++.
* ``moses_tokenized``: already-tokenized lowercase hyps vs ``target_text``,
  BLEU with ``tokenize=none``. Not comparable to RDM's published 34.49
  (no Fairseq compound-splitting).

Install::

    pip install "xlm-core[mt_eval]"
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

EvalMode = Literal["sacrebleu_detok", "moses_tokenized"]

_HYP_KEYS: Sequence[str] = ("generated_text", "text")


def _require_sacrebleu():
    try:
        import sacrebleu  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in unit test
        raise ImportError(
            "IWSLT MT eval requires sacrebleu. Install with: "
            'pip install "xlm-core[mt_eval]"'
        ) from exc
    import sacrebleu as _sb

    return _sb


@lru_cache(maxsize=2)
def _moses_detokenizer(lang: str):
    try:
        from sacremoses import MosesDetokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "IWSLT MT eval requires sacremoses. Install with: "
            'pip install "xlm-core[mt_eval]"'
        ) from exc
    return MosesDetokenizer(lang=lang)


def _detok_en(text: str) -> str:
    """Detokenize Moses-tokenized English (whitespace-split → MosesDetokenizer)."""
    text = (text or "").strip()
    if not text:
        return ""
    return _moses_detokenizer("en").detokenize(text.split())


def hypothesis_text(pred: Dict[str, Any], hyp_field: Optional[str] = None) -> str:
    if hyp_field is not None:
        return str(pred.get(hyp_field, "") or "")
    for key in _HYP_KEYS:
        if key in pred and pred[key] is not None and str(pred[key]).strip():
            return str(pred[key])
    return ""


def reference_text(
    pred: Dict[str, Any],
    mode: EvalMode,
    reference_field: Optional[str] = None,
) -> str:
    if reference_field is not None:
        return str(pred.get(reference_field, "") or "")
    if mode == "sacrebleu_detok":
        for key in ("target_raw", "truth"):
            if key in pred and pred[key] is not None and str(pred[key]).strip():
                return str(pred[key])
        return ""
    # moses_tokenized
    for key in ("target_text", "truth"):
        if key in pred and pred[key] is not None and str(pred[key]).strip():
            return str(pred[key])
    return ""


class Iwslt14MtEval:
    """Corpus MT metrics over logged IWSLT14 predictions.

    Hydra::

        post_hoc_evaluator:
          _target_: xlm.tasks.composite_eval.CompositePostHocEvaluator
          evaluators:
            prediction:
              mt:
                _target_: xlm.tasks.iwslt14_de_en.mt_eval.Iwslt14MtEval
                mode: sacrebleu_detok
    """

    def __init__(
        self,
        mode: EvalMode = "sacrebleu_detok",
        hyp_field: Optional[str] = None,
        reference_field: Optional[str] = None,
        lowercase: bool = False,
    ) -> None:
        if mode not in ("sacrebleu_detok", "moses_tokenized"):
            raise ValueError(
                f"Unknown mode {mode!r}; expected "
                "'sacrebleu_detok' or 'moses_tokenized'"
            )
        self.mode: EvalMode = mode
        self.hyp_field = hyp_field
        self.reference_field = reference_field
        self.lowercase = lowercase

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        del tokenizer, kwargs
        if not predictions:
            return predictions, {}

        sacrebleu = _require_sacrebleu()
        hyps: List[str] = []
        refs: List[str] = []

        for pred in predictions:
            hyp = hypothesis_text(pred, self.hyp_field).strip()
            ref = reference_text(pred, self.mode, self.reference_field).strip()
            if self.mode == "sacrebleu_detok":
                hyp = _detok_en(hyp)
                # target_raw is already detokenized; leave as-is
            if self.lowercase:
                hyp = hyp.lower()
                ref = ref.lower()
            pred["mt_hypothesis"] = hyp
            pred["mt_reference"] = ref
            pred["mt_eval_mode"] = self.mode
            hyps.append(hyp)
            refs.append(ref)

        aggregated: Dict[str, Any] = {}

        if self.mode == "sacrebleu_detok":
            bleu_metric = sacrebleu.metrics.BLEU()
            chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
            bleu = bleu_metric.corpus_score(hyps, [refs])
            chrf = chrf_metric.corpus_score(hyps, [refs])
            aggregated["sacrebleu"] = float(bleu.score)
            aggregated["chrfpp"] = float(chrf.score)
            # Non-numeric metadata on rows (results JSON); Lightning logs scalars only
            signature = str(bleu_metric.get_signature())
            chrf_signature = str(chrf_metric.get_signature())
            for pred in predictions:
                pred["sacrebleu_signature"] = signature
                pred["chrfpp_signature"] = chrf_signature
            logger.info(
                "Iwslt14MtEval[%s]: SacreBLEU=%.2f chrF++=%.2f (%d segs) [%s]",
                self.mode,
                bleu.score,
                chrf.score,
                len(hyps),
                signature,
            )
        else:
            bleu_metric = sacrebleu.metrics.BLEU(tokenize="none")
            bleu = bleu_metric.corpus_score(hyps, [refs])
            aggregated["tokenized_bleu"] = float(bleu.score)
            signature = str(bleu_metric.get_signature())
            for pred in predictions:
                pred["sacrebleu_signature"] = signature
            logger.info(
                "Iwslt14MtEval[%s]: tokenized BLEU=%.2f (%d segs) [%s]",
                self.mode,
                bleu.score,
                len(hyps),
                signature,
            )

        return predictions, aggregated
