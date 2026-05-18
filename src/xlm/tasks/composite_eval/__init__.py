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

For one dataloader pattern you may use a **dict of named sub-evaluators** (compose in YAML;
run order is **key order**)::

    evaluators:
      prediction:
        mauve:
          _target_: xlm.tasks.owt.mauve_text_eval.MauveTextEval
        gen_ppl:
          _target_: xlm.tasks.owt.generative_perplexity_post_hoc.GenerativePerplexityPostHocEval
          ...

A **list** of evaluators is still supported for the same pattern. A **single** evaluator
instance is unchanged. Returned ``aggregated_metrics`` dicts are merged (duplicate keys:
later sub-evaluator wins, with a warning).
"""

from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, ListConfig

from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def _normalize_to_chain(spec: Any) -> List[Any]:
    """Turn a pattern value into an ordered list of evaluator instances."""
    if isinstance(spec, (list, tuple, ListConfig)):
        return list(spec)
    if isinstance(spec, (dict, DictConfig)):
        if not spec:
            return []
        return [spec[k] for k in spec.keys()]
    return [spec]


class CompositePostHocEvaluator:
    """Routes ``eval()`` calls to task-specific evaluator(s) chosen by dataloader name.

    The ``evaluators`` dict maps a *pattern* (substring) to one evaluator instance, a
    **list** of instances, or a **dict** of name → instance (names are for structure and
    ordering only; run order follows dict / YAML key order).

    When ``eval()`` is called with a ``dataloader_name``, the first pattern that is a
    substring of the name is selected. If nothing matches, the predictions are returned
    unchanged with empty metrics.

    This is a drop-in replacement for a single evaluator: the existing
    ``Harness.compute_post_hoc_metrics`` passes ``dataloader_name`` through,
    and evaluators that don't use it simply ignore the kwarg.

    Args:
        evaluators: Mapping from dataloader-name substring to one evaluator, a list of
            evaluators, or a dict of evaluators. Each must implement
            ``eval(predictions, tokenizer=..., **kwargs)``.
    """

    def __init__(self, evaluators: Dict[str, Any]) -> None:
        self.evaluators = evaluators

    def _run_chain(
        self,
        chain: List[Any],
        predictions: List[Dict[str, Any]],
        tokenizer: Any,
        dataloader_name: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        preds = predictions
        all_metrics: Dict[str, Any] = {}
        for sub in chain:
            preds, m = sub.eval(
                preds,
                tokenizer=tokenizer,
                dataloader_name=dataloader_name,
                **kwargs,
            )
            for k, v in m.items():
                if k in all_metrics:
                    logger.warning(
                        "CompositePostHocEvaluator: duplicate metric key %r; "
                        "overwriting with value from a later evaluator in the chain.",
                        k,
                    )
                all_metrics[k] = v
        return preds, all_metrics

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
        dataloader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if dataloader_name is not None:
            for pattern, spec in self.evaluators.items():
                if pattern in dataloader_name:
                    chain = _normalize_to_chain(spec)
                    return self._run_chain(
                        chain,
                        predictions,
                        tokenizer,
                        dataloader_name,
                        kwargs,
                    )

        if len(self.evaluators) == 1:
            spec = next(iter(self.evaluators.values()))
            chain = _normalize_to_chain(spec)
            return self._run_chain(
                chain,
                predictions,
                tokenizer,
                dataloader_name,
                kwargs,
            )

        logger.warning(
            f"CompositePostHocEvaluator: no evaluator matched "
            f"dataloader_name={dataloader_name!r}. "
            f"Available patterns: {list(self.evaluators.keys())}"
        )
        return predictions, {}
