"""Composite post-hoc evaluator that routes to task-specific evaluators.

Usage in Hydra config::

    post_hoc_evaluator:
      _target_: xlm.tasks.composite_eval.CompositePostHocEvaluator
      evaluators:
        math500_prediction:
          _target_: xlm.tasks.math500.Math500Eval
        denovo_prediction:
          _target_: xlm.tasks.safe_molgen.DeNovoEval
          use_bracket_safe: true
"""

from typing import Any, Dict, List, Optional, Tuple

from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class CompositePostHocEvaluator:
    """Routes ``eval()`` calls to a task-specific evaluator chosen by
    dataloader name.

    The ``evaluators`` dict maps a *pattern* (substring) to an evaluator
    instance.  When ``eval()`` is called with a ``dataloader_name``, the first
    evaluator whose key is a substring of the name is selected.  If nothing
    matches, the predictions are returned unchanged with empty metrics.

    This is a drop-in replacement for a single evaluator: the existing
    ``Harness.compute_post_hoc_metrics`` passes ``dataloader_name`` through,
    and evaluators that don't use it simply ignore the kwarg.

    Args:
        evaluators: Mapping from dataloader-name substring to evaluator.
            Each evaluator must implement
            ``eval(predictions, tokenizer=..., **kwargs)``.
    """

    def __init__(self, evaluators: Dict[str, Any]) -> None:
        self.evaluators = evaluators

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
        dataloader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if dataloader_name is not None:
            for pattern, evaluator in self.evaluators.items():
                if pattern in dataloader_name:
                    return evaluator.eval(
                        predictions,
                        tokenizer=tokenizer,
                        dataloader_name=dataloader_name,
                        **kwargs,
                    )

        # Fallback: try the first evaluator (single-evaluator convenience) or
        # return predictions unchanged.
        if len(self.evaluators) == 1:
            evaluator = next(iter(self.evaluators.values()))
            return evaluator.eval(
                predictions,
                tokenizer=tokenizer,
                dataloader_name=dataloader_name,
                **kwargs,
            )

        logger.warning(
            f"CompositePostHocEvaluator: no evaluator matched "
            f"dataloader_name={dataloader_name!r}. "
            f"Available patterns: {list(self.evaluators.keys())}"
        )
        return predictions, {}
