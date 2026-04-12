"""HumanEval-Infill (single-line) for xlm eval / DreamOn-style runs.
"""

from __future__ import annotations

import json
import os
import tempfile
import abc
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer
from xlm.utils.rank_zero import RankedLogger
from human_eval_infilling.data import read_problems
from human_eval_infilling.evaluation import evaluate_functional_correctness
from datasets import Dataset

logger = RankedLogger(__name__, rank_zero_only=True)


def humaneval_infill_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: Any,
) -> Dict[str, Any]:
    """Tokenize prefix / suffix / middle and build a single span of mask tokens.

    Returns ``prompt_ids`` (prefix + masks + suffix) and ``input_ids`` (full
    sequence without masks) for :class:`mlm.datamodule_mlm.MLMInfillWithExactTargetPredCollator`.
    """
    prefix = example["prompt"]
    suffix = example["suffix"]
    middle = example["canonical_solution"]
    task_id = example["task_id"]

    pre_ids = tokenizer.encode(prefix)
    suf_ids = tokenizer.encode(suffix)
    mid_ids = tokenizer.encode(middle)

    return {
        "prefix_ids": pre_ids,
        "suffix_ids": suf_ids,
        "middle_ids": mid_ids,
        "task_id": task_id,
        "canonical_solution": middle,
        "prefix": prefix,
        "suffix": suffix,
    }

class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, tokens, add_bos, add_eos):
        pass

    @abc.abstractmethod
    def decode(self, tokens):
        pass

    @abc.abstractmethod
    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass
    
class HFTokenizerWrapper(Tokenizer):
    def __init__(self, hf_tokenizer: str) -> None:
        self.tokenizer = hf_tokenizer
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.mask_id = self.tokenizer.mask_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.expand_id = 151667

        self.bos_token_id = self.bos_id
        self.eos_token_id = self.eos_id
        self.mask_token_id = self.mask_id
        self.expand_token_id = self.expand_id
        self.pad_token_id = self.pad_id

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + self.tokenizer.encode(s) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: List[int], **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)
    
    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass

def get_tokenizer(pretrained_model_name_or_path: str ):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
    tokenizer = HFTokenizerWrapper(tokenizer)
    return tokenizer

def load_func(benchmark_name: str):
    problems = read_problems(benchmark_name)
    prefixs = [problems[task_id]["prompt"] for task_id in problems]
    suffixs = [problems[task_id]["suffix"] for task_id in problems]
    canonical_solutions = [problems[task_id]["canonical_solution"] for task_id in problems]
    test = [problems[task_id]["test"] for task_id in problems]
    entry_points = [problems[task_id]["entry_point"] for task_id in problems]
    data = {
        "task_id": list(problems.keys()),
        "prompt": prefixs,
        "suffix": suffixs,
        "canonical_solution": canonical_solutions,
        "test": test,
        "entry_points": entry_points,
    }
    return Dataset.from_dict(data)

class HumanEvalInfillEval:
    """Post-hoc evaluator: write HumanEval-Infill samples and optional pass@k.

    ``predictions`` entries should include ``text`` (full decoded infill line),
    ``prefix``, ``suffix``, and ``task_id`` (from ``additional_fields_from_batch``).
    """

    def __init__(self,benchmark_name):
        self.benchmark_name = benchmark_name

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        del tokenizer, kwargs
        if not predictions:
            return predictions, {}

        samples: List[Dict[str, Any]] = []
        for pred in predictions:
            full = pred.get("text", "") or ""
            prefix = pred.get("prefix", "") or ""
            suffix = pred.get("suffix", "") or ""
            task_id = pred.get("task_id", "")
            samples.append(
                {
                    "task_id": task_id,
                    "completion": full,
                    "prefix": prefix,
                    "suffix": suffix,
                    "ground_truth_middle": pred.get("canonical_solution", ""),
                }
            )

        metrics: Dict[str, Any] = {}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for row in samples:
                f.write(json.dumps(row) + "\n")
            tmp_path = f.name
        try:
            results = evaluate_functional_correctness(
                self.benchmark_name,
                tmp_path,
                [1],
                n_workers=int(os.environ.get("HUMANEVAL_WORKERS", "16")),
                timeout=float(os.environ.get("HUMANEVAL_TIMEOUT", "3.0")),
            )
            metrics = results
            logger.info("HumanEvalInfillEval: %s", results)
        except Exception as e:
            print(
                f"HumanEvalInfillEval: evaluate_functional_correctness failed: {e}"
            )
            logger.exception(
                "HumanEvalInfillEval: evaluate_functional_correctness failed"
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return predictions, metrics


def _jsonl_row_to_preprocess_example(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map a JSONL object (benchmark or prediction row) to ``humaneval_infill_preprocess_fn`` inputs."""
    prompt = row.get("prompt")
    if prompt is None:
        prompt = row.get("prefix", "") or ""
    canon = (
        row.get("canonical_solution")
        or row.get("ground_truth_middle")
        or row.get("middle")
        or ""
    )
    return {
        "task_id": row.get("task_id", ""),
        "prompt": prompt,
        "suffix": row.get("suffix", "") or "",
        "ground_truth_middle": canon,
        "completion": row.get("text", ""),
    }


def humaneval_infill_from_file(file_path: str,benchmark_name: str) -> List[Dict[str, Any]]:
    import gzip

    rows_out: List[Dict[str, Any]] = []
    if file_path.endswith(".gz"):
        fp_ctx = gzip.open(file_path, "rt", encoding="utf-8")
    else:
        fp_ctx = open(file_path, "r", encoding="utf-8")
    with fp_ctx as fp:
        for line in fp:
            raw = json.loads(line)
            example = _jsonl_row_to_preprocess_example(raw)
            rows_out.append(example)
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for row in rows_out:
                f.write(json.dumps(row) + "\n")
            tmp_path = f.name
    try:
        results = evaluate_functional_correctness(
            benchmark_name,
            tmp_path,
            [1],
            n_workers=int(os.environ.get("HUMANEVAL_WORKERS", "16")),
            timeout=float(os.environ.get("HUMANEVAL_TIMEOUT", "3.0")),
        )
        metrics = results
        logger.info("HumanEvalInfillEval: %s", results)
    except Exception as e:
        print(
            f"HumanEvalInfillEval: evaluate_functional_correctness failed: {e}"
        )
        logger.exception(
            "HumanEvalInfillEval: evaluate_functional_correctness failed"
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return metrics