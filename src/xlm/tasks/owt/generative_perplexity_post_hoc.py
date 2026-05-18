"""Post-hoc generative perplexity (judge LM) evaluation for ``Harness`` predictions.

Reads logged prediction rows (same JSONL as other post-hoc evaluators), scores
``text`` with one or more :class:`~xlm.generative_perplexity.GenerativePerplexityEvaluator`
instances, and returns per-row fields plus aggregated metrics.

Judge LMs use ``default_judge_device`` / per-evaluator overrides in this class's
config—not the training module's device.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from xlm.generative_perplexity import (
    GenerativePerplexityEvaluator,
    compute_entropy_and_length_for_sample,
    compute_nll_for_sample,
)


def _to_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


def _compute_percentage_change(new_val: float, old_val: float) -> float:
    if old_val == 0:
        return 0.0 if new_val == 0 else float("inf")
    return ((new_val - old_val) / old_val) * 100.0


class GenerativePerplexityPostHocEval:
    """Score generated text with external causal LMs (generative perplexity judges)."""

    def __init__(
        self,
        evaluators: Dict[str, GenerativePerplexityEvaluator],
        default_judge_device: str = "cuda",
        judge_devices: Optional[Dict[str, str]] = None,
        metric_prefix: str = "",
    ) -> None:
        """
        Args:
            evaluators: Names -> instantiated ``GenerativePerplexityEvaluator`` objects.
            default_judge_device: Device string for judges (e.g. ``\"cuda:1\"``, ``\"cpu\"``).
                Use ``\"auto\"`` for CUDA if available else CPU.
            judge_devices: Optional per-evaluator device overrides (name -> device string).
            metric_prefix: Optional prefix for every key in ``aggregated_metrics``.
        """
        self.evaluators = evaluators
        self.default_judge_device = default_judge_device
        self.judge_devices = judge_devices or {}
        self.metric_prefix = metric_prefix

    def _resolve_device(self, evaluator_name: str) -> torch.device:
        raw = self.judge_devices.get(
            evaluator_name, self.default_judge_device
        )
        if raw == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return torch.device(raw)

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
        dataloader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if tokenizer is None:
            raise ValueError(
                "GenerativePerplexityPostHocEval requires the training tokenizer."
            )
        has_infilling_data = bool(
            predictions
            and "raw_input" in predictions[0]
            and "truth" in predictions[0]
        )

        for pred in predictions:
            string = pred["text"]
            entropy, length = compute_entropy_and_length_for_sample(
                string, tokenizer
            )
            pred["entropy"] = entropy
            pred["length"] = length
            if has_infilling_data:
                raw_entropy, raw_length = compute_entropy_and_length_for_sample(
                    pred["raw_input"], tokenizer
                )
                pred["raw_entropy"] = raw_entropy
                pred["raw_length"] = raw_length
                truth_entropy, truth_length = (
                    compute_entropy_and_length_for_sample(
                        pred["truth"], tokenizer
                    )
                )
                pred["truth_entropy"] = truth_entropy
                pred["truth_length"] = truth_length
                pred["pct_change_raw_length"] = _compute_percentage_change(
                    length, raw_length
                )
                pred["pct_change_raw_entropy"] = _compute_percentage_change(
                    entropy, raw_entropy
                )
                pred["pct_change_truth_length"] = _compute_percentage_change(
                    length, truth_length
                )
                pred["pct_change_truth_entropy"] = _compute_percentage_change(
                    entropy, truth_entropy
                )

        data: Dict[str, List[float]] = {"length": [], "entropy": []}
        if has_infilling_data:
            data.update(
                {
                    "raw_length": [],
                    "raw_entropy": [],
                    "truth_length": [],
                    "truth_entropy": [],
                    "pct_change_raw_length": [],
                    "pct_change_raw_entropy": [],
                    "pct_change_truth_length": [],
                    "pct_change_truth_entropy": [],
                }
            )

        def add_keys(evaluator_name: str) -> None:
            data[f"nll_{evaluator_name}"] = []
            data[f"total_nll_{evaluator_name}"] = []
            data[f"length_{evaluator_name}"] = []
            data[f"entropy_{evaluator_name}"] = []
            data[f"perplexity_{evaluator_name}"] = []
            if has_infilling_data:
                data[f"raw_nll_{evaluator_name}"] = []
                data[f"raw_total_nll_{evaluator_name}"] = []
                data[f"raw_length_{evaluator_name}"] = []
                data[f"raw_entropy_{evaluator_name}"] = []
                data[f"raw_perplexity_{evaluator_name}"] = []

                data[f"truth_nll_{evaluator_name}"] = []
                data[f"truth_total_nll_{evaluator_name}"] = []
                data[f"truth_length_{evaluator_name}"] = []
                data[f"truth_entropy_{evaluator_name}"] = []
                data[f"truth_perplexity_{evaluator_name}"] = []

                data[f"pct_change_raw_nll_{evaluator_name}"] = []
                data[f"pct_change_raw_perplexity_{evaluator_name}"] = []
                data[f"pct_change_raw_length_{evaluator_name}"] = []
                data[f"pct_change_raw_entropy_{evaluator_name}"] = []
                data[f"pct_change_truth_nll_{evaluator_name}"] = []
                data[f"pct_change_truth_perplexity_{evaluator_name}"] = []
                data[f"pct_change_truth_length_{evaluator_name}"] = []
                data[f"pct_change_truth_entropy_{evaluator_name}"] = []

        for evaluator_name in self.evaluators:
            add_keys(evaluator_name)

        for pred in predictions:
            data["entropy"].append(float(pred["entropy"]))
            data["length"].append(float(pred["length"]))
            if has_infilling_data:
                data["raw_entropy"].append(float(pred["raw_entropy"]))
                data["raw_length"].append(float(pred["raw_length"]))
                data["truth_entropy"].append(float(pred["truth_entropy"]))
                data["truth_length"].append(float(pred["truth_length"]))
                data["pct_change_raw_length"].append(
                    float(pred["pct_change_raw_length"])
                )
                data["pct_change_raw_entropy"].append(
                    float(pred["pct_change_raw_entropy"])
                )
                data["pct_change_truth_length"].append(
                    float(pred["pct_change_truth_length"])
                )
                data["pct_change_truth_entropy"].append(
                    float(pred["pct_change_truth_entropy"])
                )

        with torch.no_grad():
            for evaluator_name, evaluator in self.evaluators.items():
                judge_device = self._resolve_device(evaluator_name)
                with evaluator.loaded(tokenizer, judge_device):
                    for pred in predictions:
                        string = pred["text"]
                        temp = compute_nll_for_sample(string, evaluator)
                        if temp is None:
                            continue
                        nll, _len, entropy_j = temp
                        pred[f"total_nll_{evaluator_name}"] = nll
                        pred[f"total_length_{evaluator_name}"] = _len
                        pred[f"entropy_{evaluator_name}"] = entropy_j
                        mean_nll = nll / _len
                        pred[f"nll_{evaluator_name}"] = mean_nll
                        perplexity = _to_float(
                            torch.exp(torch.tensor(mean_nll))
                        )
                        pred[f"perplexity_{evaluator_name}"] = perplexity
                        data[f"nll_{evaluator_name}"].append(mean_nll)
                        data[f"total_nll_{evaluator_name}"].append(nll)
                        data[f"length_{evaluator_name}"].append(float(_len))
                        data[f"entropy_{evaluator_name}"].append(
                            float(entropy_j)
                        )
                        data[f"perplexity_{evaluator_name}"].append(perplexity)

                        if has_infilling_data:
                            raw_temp = compute_nll_for_sample(
                                pred["raw_input"], evaluator
                            )
                            if raw_temp is not None:
                                raw_nll, raw_len, raw_entropy_j = raw_temp
                                pred[f"raw_total_nll_{evaluator_name}"] = (
                                    raw_nll
                                )
                                pred[f"raw_total_length_{evaluator_name}"] = (
                                    raw_len
                                )
                                pred[f"raw_entropy_{evaluator_name}"] = (
                                    raw_entropy_j
                                )
                                raw_mean_nll = raw_nll / raw_len
                                pred[f"raw_nll_{evaluator_name}"] = (
                                    raw_mean_nll
                                )
                                raw_perplexity = _to_float(
                                    torch.exp(torch.tensor(raw_mean_nll))
                                )
                                pred[f"raw_perplexity_{evaluator_name}"] = (
                                    raw_perplexity
                                )
                                pct_change_nll_raw = _compute_percentage_change(
                                    mean_nll, raw_mean_nll
                                )
                                pct_change_perplexity_raw = (
                                    _compute_percentage_change(
                                        perplexity, raw_perplexity
                                    )
                                )
                                pct_change_length_raw = (
                                    _compute_percentage_change(_len, raw_len)
                                )
                                pct_change_entropy_raw = (
                                    _compute_percentage_change(
                                        entropy_j, raw_entropy_j
                                    )
                                )
                                pred[
                                    f"pct_change_raw_nll_{evaluator_name}"
                                ] = pct_change_nll_raw
                                pred[
                                    f"pct_change_raw_perplexity_{evaluator_name}"
                                ] = pct_change_perplexity_raw
                                pred[
                                    f"pct_change_raw_length_{evaluator_name}"
                                ] = pct_change_length_raw
                                pred[
                                    f"pct_change_raw_entropy_{evaluator_name}"
                                ] = pct_change_entropy_raw

                                data[f"raw_nll_{evaluator_name}"].append(
                                    raw_mean_nll
                                )
                                data[f"raw_total_nll_{evaluator_name}"].append(
                                    raw_nll
                                )
                                data[f"raw_length_{evaluator_name}"].append(
                                    float(raw_len)
                                )
                                data[f"raw_entropy_{evaluator_name}"].append(
                                    float(raw_entropy_j)
                                )
                                data[f"raw_perplexity_{evaluator_name}"].append(
                                    raw_perplexity
                                )
                                data[
                                    f"pct_change_raw_nll_{evaluator_name}"
                                ].append(pct_change_nll_raw)
                                data[
                                    f"pct_change_raw_perplexity_{evaluator_name}"
                                ].append(pct_change_perplexity_raw)
                                data[
                                    f"pct_change_raw_length_{evaluator_name}"
                                ].append(pct_change_length_raw)
                                data[
                                    f"pct_change_raw_entropy_{evaluator_name}"
                                ].append(pct_change_entropy_raw)

                            truth_temp = compute_nll_for_sample(
                                pred["truth"], evaluator
                            )
                            if truth_temp is not None:
                                truth_nll, truth_len, truth_entropy_j = (
                                    truth_temp
                                )
                                pred[f"truth_total_nll_{evaluator_name}"] = (
                                    truth_nll
                                )
                                pred[
                                    f"truth_total_length_{evaluator_name}"
                                ] = truth_len
                                pred[f"truth_entropy_{evaluator_name}"] = (
                                    truth_entropy_j
                                )
                                truth_mean_nll = truth_nll / truth_len
                                pred[f"truth_nll_{evaluator_name}"] = (
                                    truth_mean_nll
                                )
                                truth_perplexity = _to_float(
                                    torch.exp(torch.tensor(truth_mean_nll))
                                )
                                pred[f"truth_perplexity_{evaluator_name}"] = (
                                    truth_perplexity
                                )
                                pct_change_nll_truth = (
                                    _compute_percentage_change(
                                        mean_nll, truth_mean_nll
                                    )
                                )
                                pct_change_perplexity_truth = (
                                    _compute_percentage_change(
                                        perplexity, truth_perplexity
                                    )
                                )
                                pct_change_length_truth = (
                                    _compute_percentage_change(_len, truth_len)
                                )
                                pct_change_entropy_truth = (
                                    _compute_percentage_change(
                                        entropy_j, truth_entropy_j
                                    )
                                )
                                pred[
                                    f"pct_change_truth_nll_{evaluator_name}"
                                ] = pct_change_nll_truth
                                pred[
                                    f"pct_change_truth_perplexity_{evaluator_name}"
                                ] = pct_change_perplexity_truth
                                pred[
                                    f"pct_change_truth_length_{evaluator_name}"
                                ] = pct_change_length_truth
                                pred[
                                    f"pct_change_truth_entropy_{evaluator_name}"
                                ] = pct_change_entropy_truth

                                data[f"truth_nll_{evaluator_name}"].append(
                                    truth_mean_nll
                                )
                                data[
                                    f"truth_total_nll_{evaluator_name}"
                                ].append(truth_nll)
                                data[f"truth_length_{evaluator_name}"].append(
                                    float(truth_len)
                                )
                                data[f"truth_entropy_{evaluator_name}"].append(
                                    float(truth_entropy_j)
                                )
                                data[
                                    f"truth_perplexity_{evaluator_name}"
                                ].append(truth_perplexity)
                                data[
                                    f"pct_change_truth_nll_{evaluator_name}"
                                ].append(pct_change_nll_truth)
                                data[
                                    f"pct_change_truth_perplexity_{evaluator_name}"
                                ].append(pct_change_perplexity_truth)
                                data[
                                    f"pct_change_truth_length_{evaluator_name}"
                                ].append(pct_change_length_truth)
                                data[
                                    f"pct_change_truth_entropy_{evaluator_name}"
                                ].append(pct_change_entropy_truth)

        aggregated: Dict[str, float] = {}
        for key, values in data.items():
            if not values:
                continue
            if (
                not key.startswith("total_nll_")
                and not key.startswith("raw_total_nll_")
                and not key.startswith("truth_total_nll_")
            ):
                aggregated[key] = _to_float(
                    torch.tensor(values).double().mean()
                )
            elif key.startswith("total_nll_"):
                ename = re.sub(r"total_nll_", "", key)
                total_length = torch.tensor(
                    data[f"length_{ename}"], dtype=torch.float64
                ).sum()
                total_nll = torch.tensor(values, dtype=torch.float64).sum()
                perplexity = _to_float(torch.exp(total_nll / total_length))
                aggregated[f"total_perplexity_{ename}"] = perplexity
            elif key.startswith("raw_total_nll_"):
                ename = re.sub(r"raw_total_nll_", "", key)
                total_length = torch.tensor(
                    data[f"raw_length_{ename}"], dtype=torch.float64
                ).sum()
                total_nll = torch.tensor(values, dtype=torch.float64).sum()
                perplexity = _to_float(torch.exp(total_nll / total_length))
                aggregated[f"raw_total_perplexity_{ename}"] = perplexity
            elif key.startswith("truth_total_nll_"):
                ename = re.sub(r"truth_total_nll_", "", key)
                total_length = torch.tensor(
                    data[f"truth_length_{ename}"], dtype=torch.float64
                ).sum()
                total_nll = torch.tensor(values, dtype=torch.float64).sum()
                perplexity = _to_float(torch.exp(total_nll / total_length))
                aggregated[f"truth_total_perplexity_{ename}"] = perplexity

        if self.metric_prefix:
            aggregated = {
                f"{self.metric_prefix}{k}": v for k, v in aggregated.items()
            }

        return predictions, aggregated
