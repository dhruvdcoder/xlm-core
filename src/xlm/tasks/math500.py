"""MATH-500 evaluation task.

Provides:
- ``Math500Eval``: post-hoc evaluator for the MATH-500 benchmark.
- ``math500_preprocess_fn``: dataset map function that constructs fewshot
  prompts and tokenizes them for use as a prediction dataloader.

Dependencies (install when using this module):
    pip install math_verify
"""

from typing import Any, Dict, List, Optional, Tuple

from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


# ---------------------------------------------------------------------------
# Post-hoc evaluator
# ---------------------------------------------------------------------------


class Math500Eval:
    """Post-hoc evaluator for the MATH-500 benchmark.

    Reads predictions with ``text`` (model generation) and ``truth`` (gold
    answer), extracts mathematical expressions from both, and checks
    equivalence using the ``math_verify`` library.

    Uses the same verification logic as ``prd2``'s
    ``math_verify_utils.process_results``: ``math_verify.parse`` to extract
    a structured answer, then ``math_verify.verify`` to check equivalence.

    Hydra config example::

        post_hoc_evaluator:
          _target_: xlm.tasks.math500.Math500Eval

    Or inside a ``CompositePostHocEvaluator``::

        post_hoc_evaluator:
          _target_: xlm.tasks.composite_eval.CompositePostHocEvaluator
          evaluators:
            math500_prediction:
              _target_: xlm.tasks.math500.Math500Eval
    """

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Score each prediction against the gold answer.

        Args:
            predictions: List of dicts, each containing at least:
                - ``text``: raw model output string
                - ``truth``: gold answer string (e.g. ``"$42$"``)
            tokenizer: Unused, kept for interface consistency.

        Returns:
            ``(predictions, aggregated_metrics)`` — predictions are updated
            in-place with ``parsed_answer``, ``parsed_gold``, and ``correct``
            fields.
        """
        from math_verify import parse, verify

        if not predictions:
            return predictions, {}

        n_correct = 0
        n_answer_parse_failures = 0
        n_gold_parse_failures = 0
        n_verify_failures = 0
        for pred in predictions:
            raw_text = pred.get("text", "")
            gold_text = pred.get("truth", "") or pred.get("answer", "")

            answer_parsed = True
            try:
                parsed_answer = parse(raw_text)
            except Exception:
                parsed_answer = None
                answer_parsed = False
                n_answer_parse_failures += 1

            gold_parsed = True
            try:
                parsed_gold = parse(gold_text)
            except Exception:
                parsed_gold = None
                gold_parsed = False
                n_gold_parse_failures += 1

            pred["parsed_answer"] = str(parsed_answer)
            pred["parsed_gold"] = str(parsed_gold)
            pred["answer_parsed"] = answer_parsed
            pred["gold_parsed"] = gold_parsed

            if parsed_answer is not None and parsed_gold is not None:
                try:
                    correct = bool(verify(parsed_gold, parsed_answer))
                except Exception:
                    correct = False
                    n_verify_failures += 1
            else:
                correct = False

            pred["correct"] = correct
            n_correct += int(correct)

        total = len(predictions)
        accuracy = n_correct / total

        logger.info(
            "Math500Eval: %d/%d correct (%.2f%%) | "
            "parse failures: answer=%d, gold=%d | "
            "verify failures: %d",
            n_correct,
            total,
            accuracy * 100,
            n_answer_parse_failures,
            n_gold_parse_failures,
            n_verify_failures,
        )

        all_time = [p["time_taken"] for p in predictions if "time_taken" in p]
        all_steps = [
            p["steps_taken"] for p in predictions if "steps_taken" in p
        ]

        aggregated_metrics = {
            "math_equal_at_1": accuracy,
            "n_answer_parse_failures": n_answer_parse_failures,
            "n_gold_parse_failures": n_gold_parse_failures,
            "n_verify_failures": n_verify_failures,
        }
        if all_time:
            aggregated_metrics["avg_time_taken"] = sum(all_time) / len(
                all_time
            )
        if all_steps:
            aggregated_metrics["avg_steps_taken"] = sum(all_steps) / len(
                all_steps
            )
        return predictions, aggregated_metrics


# ---------------------------------------------------------------------------
# Dataset preprocessing (prompt construction + tokenization)
# ---------------------------------------------------------------------------

_FEWSHOT_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _load_fewshot_examples(
    dataset_path: str = "HuggingFaceH4/MATH-500",
    split: str = "train",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load and cache fewshot examples from the dataset.

    Returns all examples from the split; the caller selects a subset.
    """
    cache_key = f"{dataset_path}:{split}:{seed}"
    if cache_key not in _FEWSHOT_CACHE:
        import datasets as hf_datasets

        ds = hf_datasets.load_dataset(dataset_path, split=split)
        ds = ds.shuffle(seed=seed)
        _FEWSHOT_CACHE[cache_key] = list(ds)
    return _FEWSHOT_CACHE[cache_key]


def math500_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: Any,
    *,
    num_fewshot: int = 4,
    dataset_path: str = "HuggingFaceH4/MATH-500",
    fewshot_split: str = "test",
    fewshot_seed: int = 42,
    block_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a fewshot prompt for a MATH-500 example and tokenize it.

    Intended for use as a HuggingFace ``datasets.Dataset.map`` function via
    the xlm-core datamodule ``preprocess_function`` / ``on_the_fly_processor``
    config knob.

    The prompt format matches the lm-evaluation-harness yaml for MATH-500
    (``discrete-diffusion/src/dd/tasks/math500/math500.yaml``)::

        Problem: {problem_1}
        Answer:{solution_1}

        ...

        Problem: {problem_N}
        Answer:{solution_N}

        Problem: {current_problem}
        Answer:

    Gold answer is stored in ``answer`` so that ``LogPredictions`` can carry
    it through to the predictions JSONL via ``additional_fields_from_batch``,
    where ``Math500Eval`` picks it up as ``truth``.

    Args:
        example: A single dataset row with ``problem``, ``solution``, and
            ``answer`` fields.
        tokenizer: Tokenizer with an ``encode`` method.
        num_fewshot: Number of fewshot examples to prepend.
        dataset_path: HuggingFace dataset path for fewshot examples.
        fewshot_split: Split to draw fewshot examples from.
        fewshot_seed: Seed for shuffling fewshot pool.
        block_size: If set, truncate ``prompt_ids`` to this length.

    Returns:
        Dict with ``prompt_ids`` (prompt), ``target_ids`` (suffix; empty for
        prompt-only prediction), and ``answer`` (str).
    Ref:
      https://github.com/dhruvdcoder/discrete-diffusion/src/dd/tasks/math500/math_verify_utils.py
    """
    fewshot_pool = _load_fewshot_examples(
        dataset_path, fewshot_split, fewshot_seed
    )

    current_problem = example["problem"]

    # Select fewshot examples, excluding the current one
    fewshot_examples: List[Dict[str, Any]] = []
    for ex in fewshot_pool:
        if ex["problem"] == current_problem:
            continue
        fewshot_examples.append(ex)
        if len(fewshot_examples) >= num_fewshot:
            break

    # Build prompt
    parts: List[str] = []
    for ex in fewshot_examples:
        parts.append(f"Problem: {ex['problem']}\nAnswer:{ex['solution']}")
    parts.append(f"Problem: {current_problem}\nAnswer:")
    prompt = "\n\n".join(parts)

    raw_answer = example.get("answer", "")
    answer = "$" + str(raw_answer) + "$"

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if block_size is not None:
        prompt_ids = prompt_ids[:block_size]

    return {
        "prompt_ids": prompt_ids,
        "target_ids": [],
        "answer": answer,
    }
