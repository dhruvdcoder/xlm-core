"""Tests for IWSLT14 MT post-hoc evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from xlm.tasks.iwslt14_de_en.mt_eval import (
    Iwslt14MtEval,
    hypothesis_text,
    reference_text,
)


pytest.importorskip("sacrebleu")
pytest.importorskip("sacremoses")


def _rows(
    hyps: List[str],
    *,
    target_raw: Optional[List[str]] = None,
    target_text: Optional[List[str]] = None,
    use_generated: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, hyp in enumerate(hyps):
        row: Dict[str, Any] = {"text": f"PREFIX {hyp}"}
        if use_generated:
            row["generated_text"] = hyp
        if target_raw is not None:
            row["target_raw"] = target_raw[i]
        if target_text is not None:
            row["target_text"] = target_text[i]
        out.append(row)
    return out


def test_hypothesis_prefers_generated_text() -> None:
    pred = {"text": "full prefix suffix", "generated_text": "suffix only"}
    assert hypothesis_text(pred) == "suffix only"


def test_hypothesis_falls_back_to_text() -> None:
    assert hypothesis_text({"text": "only text"}) == "only text"


def test_reference_by_mode() -> None:
    pred = {"target_raw": "Hello world.", "target_text": "hello world .", "truth": "x"}
    assert reference_text(pred, "sacrebleu_detok") == "Hello world."
    assert reference_text(pred, "moses_tokenized") == "hello world ."


def test_sacrebleu_detok_perfect_match() -> None:
    refs = ["Hello world.", "This is a test."]
    # Moses-tokenized-looking hyps (as CharBPE decode of training targets would look)
    hyps = ["Hello world .", "This is a test ."]
    rows = _rows(hyps, target_raw=refs)
    preds, metrics = Iwslt14MtEval(mode="sacrebleu_detok").eval(rows)
    assert metrics["sacrebleu"] == pytest.approx(100.0, abs=1e-3)
    assert metrics["chrfpp"] > 99.0
    assert preds[0]["sacrebleu_signature"]
    assert "tok:13a" in preds[0]["sacrebleu_signature"]
    assert preds[0]["mt_eval_mode"] == "sacrebleu_detok"


def test_moses_tokenized_perfect_match() -> None:
    toks = ["hello world .", "this is a test ."]
    rows = _rows(toks, target_text=toks)
    _, metrics = Iwslt14MtEval(mode="moses_tokenized").eval(rows)
    assert metrics["tokenized_bleu"] == pytest.approx(100.0, abs=1e-3)
    assert "sacrebleu" not in metrics


def test_moses_tokenized_signature_uses_none() -> None:
    toks = ["a b c"]
    rows = _rows(toks, target_text=toks)
    preds, _ = Iwslt14MtEval(mode="moses_tokenized").eval(rows)
    assert "tok:none" in preds[0]["sacrebleu_signature"]


def test_empty_predictions() -> None:
    preds, metrics = Iwslt14MtEval().eval([])
    assert preds == []
    assert metrics == {}
