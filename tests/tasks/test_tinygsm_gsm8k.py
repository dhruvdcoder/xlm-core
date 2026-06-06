"""Tests for TinyGSM GSM8K preprocessing and code-execution eval."""

from typing import Any, Dict, List

import pytest

from xlm.tasks.tinygsm.gsm8k import (
    Gsm8kCodeEval,
    evaluate_samples,
    extract_gsm8k_final_answer,
    gold_answer_from_tinygsm_code,
    gsm8k_preprocess_fn,
    prediction_code_text,
    tinygsm_pred_preprocess_fn,
)


class _MockTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        return [ord(c) for c in text]


def test_extract_gsm8k_final_answer_hash_marker() -> None:
    assert extract_gsm8k_final_answer("Reasoning...\n#### 72") == "72"


def test_extract_gsm8k_final_answer_commas_and_negative() -> None:
    assert extract_gsm8k_final_answer("#### -1,234") == "-1234"


def test_extract_gsm8k_final_answer_fallback_last_number() -> None:
    assert extract_gsm8k_final_answer("The answer is 99 and also 100") == "100"


def test_extract_gsm8k_final_answer_empty() -> None:
    assert extract_gsm8k_final_answer("no numbers here") == ""


def test_gsm8k_preprocess_fn() -> None:
    tok = _MockTokenizer()
    row: Dict[str, Any] = {
        "question": "How many apples?",
        "answer": "Step by step.\n#### 42",
    }
    out = gsm8k_preprocess_fn(row, tok, sep="\n")
    assert out["answer"] == "42"
    assert out["input_token_ids"] == []
    assert out["prompt_token_ids"] == tok.encode("How many apples?") + tok.encode("\n")


def test_gold_answer_from_tinygsm_code() -> None:
    code = "def simple_math_problem():\n    return 42\n"
    assert gold_answer_from_tinygsm_code(code) == "42"


def test_tinygsm_pred_preprocess_fn() -> None:
    tok = _MockTokenizer()
    row: Dict[str, Any] = {
        "question": "Q?",
        "code": "def simple_math_problem():\n    return 7\n",
    }
    out = tinygsm_pred_preprocess_fn(row, tok, sep="\n")
    assert out["answer"] == "7"
    assert out["input_token_ids"] == []
    assert out["prompt_token_ids"] == tok.encode("Q?") + tok.encode("\n")


def test_evaluate_samples_valid_code() -> None:
    code = "def simple_math_problem():\n    return 42\n"
    assert evaluate_samples(code, "42")


def test_evaluate_samples_wrong_answer() -> None:
    code = "def simple_math_problem():\n    return 41\n"
    assert not evaluate_samples(code, "42")


def test_evaluate_samples_missing_function() -> None:
    assert not evaluate_samples("x = 1", "1")


def test_gsm8k_code_eval_aggregates() -> None:
    evaluator = Gsm8kCodeEval(timeout_s=1.0)
    preds = [
        {
            "text": "def simple_math_problem():\n    return 10\n",
            "answer": "10",
        },
        {
            "text": "def simple_math_problem():\n    return 9\n",
            "answer": "10",
        },
    ]
    updated, metrics = evaluator.eval(preds)
    assert len(updated) == 2
    assert updated[0]["correct"] is True
    assert updated[1]["correct"] is False
    assert metrics["code_exec_accuracy"] == 0.5
    assert metrics["n_correct"] == 1
    assert metrics["n_total"] == 2


def test_gsm8k_code_eval_truth_field() -> None:
    evaluator = Gsm8kCodeEval()
    preds = [
        {
            "text": "def simple_math_problem():\n    return 7\n",
            "truth": "7",
        },
    ]
    _, metrics = evaluator.eval(preds)
    assert metrics["code_exec_accuracy"] == 1.0


def test_prediction_code_text_prefers_generated_text() -> None:
    assert (
        prediction_code_text(
            {
                "text": "question\n",
                "generated_text": "def simple_math_problem():\n    return 1\n",
            }
        )
        == "def simple_math_problem():\n    return 1\n"
    )


def test_prediction_code_text_falls_back_to_text() -> None:
    assert prediction_code_text({"text": "only full"}) == "only full"


def test_gsm8k_code_eval_uses_generated_text() -> None:
    evaluator = Gsm8kCodeEval()
    preds = [
        {
            "text": "ignore this prefix",
            "generated_text": "def simple_math_problem():\n    return 3\n",
            "answer": "3",
        },
    ]
    _, metrics = evaluator.eval(preds)
    assert metrics["code_exec_accuracy"] == 1.0


def test_gsm8k_code_eval_empty_predictions() -> None:
    evaluator = Gsm8kCodeEval()
    preds, metrics = evaluator.eval([])
    assert preds == []
    assert metrics == {}
