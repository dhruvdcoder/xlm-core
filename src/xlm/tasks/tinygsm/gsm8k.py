"""GSM8K test evaluation for TinyGSM-trained code generators.

Preprocessing mirrors TinyGSM seq2seq layout but uses GSM8K fields (``question``,
``answer``) and leaves the suffix empty at inference. Scoring executes generated
Python and compares numeric results, following PUMA's gsm8k_eval.py:
https://github.com/JaeyeonKim01/PUMA/blob/main/eval/gsm8k_eval.py
"""

from __future__ import annotations

import contextlib
import math
import re
import signal
import warnings
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

_ANS_RE = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")


def extract_gsm8k_final_answer(ans_text: str) -> str:
    """Extract the numeric final answer from a GSM8K ``answer`` field.

    GSM8K answers end with ``#### 72``. Falls back to the last number in the
    string if the marker is missing.
    """
    m = _ANS_RE.search(ans_text)
    if not m:
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", ans_text)
        return nums[-1].replace(",", "") if nums else ""
    return m.group(1).replace(",", "")


def gsm8k_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    sep: str = "\n",
) -> Dict[str, Any]:
    """Tokenize GSM8K test rows for seq2seq MDM prediction.

    Args:
        example: HF row with ``question`` and ``answer`` fields.
        tokenizer: Hugging Face tokenizer (``encode``, no special tokens).
        sep: String between question and generated code region (PUMA default).

    Returns:
        Updated example with ``prompt_token_ids``, empty ``input_token_ids``,
        and ``answer`` set to the numeric gold string.
    """
    question = (example.get("question") or "").strip()
    raw_answer = example.get("answer") or ""
    gold = extract_gsm8k_final_answer(raw_answer)
    sep_ids = tokenizer.encode(sep, add_special_tokens=False)
    p_ids = tokenizer.encode(question, add_special_tokens=False)
    example["prompt_token_ids"] = p_ids + sep_ids
    example["input_token_ids"] = []
    example["answer"] = gold
    return example


def execute_tinygsm_code(
    code: str, timeout_s: float = 1.0
) -> Optional[int | float]:
    """Run reference TinyGSM ``code`` and return ``simple_math_problem()`` value."""
    extracted = _extract_code(code)
    try:
        with _time_limit(timeout_s):
            ns = _safe_exec_no_timer(extracted)
            fn = ns.get("simple_math_problem", None)
            if fn is None:
                return None
            return _to_number(fn())
    except (_Timeout, Exception):
        return None


def gold_answer_from_tinygsm_code(code: str, timeout_s: float = 1.0) -> str:
    """Numeric gold string for post-hoc eval (empty if reference code fails)."""
    val = execute_tinygsm_code(code, timeout_s=timeout_s)
    if val is None:
        return ""
    if isinstance(val, int):
        return str(val)
    return str(val)


def tinygsm_pred_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    sep: str = "\n",
    gold_timeout_s: float = 1.0,
) -> Dict[str, Any]:
    """TinyGSM rows for seq2seq prediction: question prefix, empty suffix, numeric gold.

    Gold is computed once by executing the reference ``code`` (PUMA/TinyGSM convention).
    """
    question = (example.get("question") or "").strip()
    code = (example.get("code") or "").strip()
    gold = gold_answer_from_tinygsm_code(code, timeout_s=gold_timeout_s)
    sep_ids = tokenizer.encode(sep, add_special_tokens=False)
    p_ids = tokenizer.encode(question, add_special_tokens=False)
    example["prompt_token_ids"] = p_ids + sep_ids
    example["input_token_ids"] = []
    example["answer"] = gold
    return example


# ---------------------------------------------------------------------------
# Code execution scoring (ported from PUMA gsm8k_eval.py)
# ---------------------------------------------------------------------------


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame) -> None:
    raise _Timeout()


@contextlib.contextmanager
def _time_limit(timeout_s: float):
    """Hard wall-clock limit via SIGALRM (POSIX main thread only)."""
    has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    old_handler = None
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def _safe_exec_no_timer(code: str) -> Dict[str, Any]:
    """Execute code in a restricted namespace (no timeout; wrap with _time_limit)."""
    import math as _math

    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "round": round,
        "print": lambda *args, **kwargs: None,
    }

    def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "math":
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Import blocked: {name}")

    safe_builtins["__import__"] = _limited_import

    ns: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "math": _math,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(code, ns, ns)

    return ns


def _extract_code(text: str) -> str:
    """Heuristically extract executable Python from model output."""
    for stopper in ["<|endoftext|>", "<|eot_id|>", "</s>"]:
        if stopper in text:
            text = text.split(stopper, 1)[0]

    fence = re.search(
        r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if fence:
        text = fence.group(1)

    i = text.find("def ")
    if i != -1:
        text = text[i:]

    text = text.strip()

    lines = text.splitlines()
    for k in range(0, min(50, len(lines))):
        candidate = "\n".join(lines[: len(lines) - k]).strip()
        if not candidate:
            continue
        try:
            compile(candidate, "<sample>", "exec")
            return candidate
        except SyntaxError:
            continue

    return text


def _numbers_equal(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    if isinstance(pred, float) or isinstance(gold, float):
        return abs(float(pred) - float(gold)) <= 1e-3
    return int(pred) == int(gold)


def _to_number(x) -> Optional[int | float]:
    """Normalize return values to int or float where possible."""
    if x is None:
        return None
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        if not math.isfinite(float(x)):
            return None
        xf = float(x)
        if abs(xf - round(xf)) < 1e-6:
            return int(round(xf))
        return xf
    if isinstance(x, str):
        m = re.search(r"[-+]?\d[\d,]*\.?\d*", x)
        if not m:
            return None
        s = m.group(0).replace(",", "")
        if s.count(".") == 1:
            f = float(s)
            if abs(f - round(f)) < 1e-6:
                return int(round(f))
            return f
        return int(s)
    return None


def evaluate_samples(sample: str, answer: str, timeout_s: float = 1.0) -> bool:
    """Return True if executing ``sample`` yields the gold numeric ``answer``."""
    code = _extract_code(sample)

    try:
        with _time_limit(timeout_s):
            ns = _safe_exec_no_timer(code)
            fn = ns.get("simple_math_problem", None)
            if fn is None:
                return False
            out = fn()
    except (_Timeout, Exception):
        return False

    pred = _to_number(out)
    gold = _to_number(answer)
    return _numbers_equal(pred, gold)


def prediction_code_text(pred: Dict[str, Any]) -> str:
    """Return model output text to execute for code-exec scoring.

    Prefers ``generated_text`` (suffix-only decode from FlexMDM). Falls back to
    ``text`` (full sequence including the question prefix) for older logs.
    """
    generated = pred.get("generated_text")
    if generated is not None and str(generated).strip():
        return str(generated)
    return str(pred.get("text", ""))


def evaluate_sample_with_details(
    sample: str, answer: str, timeout_s: float = 1.0
) -> Tuple[bool, Optional[int | float], Optional[str]]:
    """Like ``evaluate_samples`` but returns (correct, pred_value, error)."""
    code = _extract_code(sample)

    try:
        with _time_limit(timeout_s):
            ns = _safe_exec_no_timer(code)
            fn = ns.get("simple_math_problem", None)
            if fn is None:
                return False, None, "simple_math_problem not defined"
            out = fn()
    except _Timeout:
        return False, None, "timeout"
    except Exception as e:
        return False, None, repr(e)

    pred = _to_number(out)
    gold = _to_number(answer)
    return _numbers_equal(pred, gold), pred, None


class Gsm8kCodeEval:
    """Post-hoc evaluator: execute generated code and compare to GSM8K gold.

    Expects prediction rows with ``generated_text`` (suffix-only decode; preferred)
    or ``text`` (full sequence), plus ``answer`` or ``truth`` (numeric gold).

    Hydra::

        post_hoc_evaluator:
          _target_: xlm.tasks.tinygsm.Gsm8kCodeEval
    """

    def __init__(self, timeout_s: float = 1.0) -> None:
        self.timeout_s = timeout_s

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not predictions:
            return predictions, {}

        n_correct = 0
        n_exec_failures = 0

        for pred in predictions:
            raw_text = prediction_code_text(pred)
            gold_text = pred.get("answer", "") or pred.get("truth", "")

            correct, pred_value, err = evaluate_sample_with_details(
                raw_text, str(gold_text), timeout_s=self.timeout_s
            )
            pred["correct"] = correct
            pred["pred_value"] = pred_value
            if err is not None:
                pred["exec_error"] = err
                n_exec_failures += 1

            n_correct += int(correct)

        total = len(predictions)
        accuracy = n_correct / total

        logger.info(
            "Gsm8kCodeEval: %d/%d correct (%.2f%%) | exec failures: %d",
            n_correct,
            total,
            accuracy * 100,
            n_exec_failures,
        )

        aggregated_metrics = {
            "code_exec_accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": total,
            "n_exec_failures": n_exec_failures,
        }
        return predictions, aggregated_metrics
