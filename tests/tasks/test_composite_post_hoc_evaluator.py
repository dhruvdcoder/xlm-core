from typing import Any, Dict, List, Optional, Tuple

from xlm.tasks.composite_eval import CompositePostHocEvaluator


def test_composite_dict_chain_runs_in_key_order_merges_metrics() -> None:
    class A:
        def eval(
            self,
            predictions: List[Dict[str, Any]],
            tokenizer: Any = None,
            dataloader_name: Optional[str] = None,
            **kwargs: Any,
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            for p in predictions:
                p["a"] = 1
            return predictions, {"m_a": 1.0}

    class B:
        def eval(
            self,
            predictions: List[Dict[str, Any]],
            tokenizer: Any = None,
            dataloader_name: Optional[str] = None,
            **kwargs: Any,
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            for p in predictions:
                p["b"] = 2
            return predictions, {"m_b": 2.0}

    comp = CompositePostHocEvaluator(
        evaluators={"prediction": {"first": A(), "second": B()}}
    )
    preds, metrics = comp.eval(
        [{"text": "x"}],
        tokenizer=None,
        dataloader_name="val/unconditional_prediction",
    )
    assert preds[0]["a"] == 1
    assert preds[0]["b"] == 2
    assert metrics == {"m_a": 1.0, "m_b": 2.0}


def test_composite_list_chain_runs_sequential_merges_metrics() -> None:
    class A:
        def eval(
            self,
            predictions: List[Dict[str, Any]],
            tokenizer: Any = None,
            dataloader_name: Optional[str] = None,
            **kwargs: Any,
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            for p in predictions:
                p["a"] = 1
            return predictions, {"m_a": 1.0}

    class B:
        def eval(
            self,
            predictions: List[Dict[str, Any]],
            tokenizer: Any = None,
            dataloader_name: Optional[str] = None,
            **kwargs: Any,
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            for p in predictions:
                p["b"] = 2
            return predictions, {"m_b": 2.0}

    comp = CompositePostHocEvaluator(evaluators={"prediction": [A(), B()]})
    preds, metrics = comp.eval(
        [{"text": "x"}],
        tokenizer=None,
        dataloader_name="val/unconditional_prediction",
    )
    assert preds[0]["a"] == 1
    assert preds[0]["b"] == 2
    assert metrics == {"m_a": 1.0, "m_b": 2.0}


def test_composite_empty_dict_pattern_runs_no_sub_evaluators() -> None:
    comp = CompositePostHocEvaluator(evaluators={"prediction": {}})
    preds, metrics = comp.eval(
        [{"text": "x"}],
        dataloader_name="val/unconditional_prediction",
    )
    assert preds == [{"text": "x"}]
    assert metrics == {}


def test_composite_single_evaluator_backward_compatible() -> None:
    class Only:
        def eval(
            self,
            predictions: List[Dict[str, Any]],
            tokenizer: Any = None,
            dataloader_name: Optional[str] = None,
            **kwargs: Any,
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            return predictions, {"ok": 3.0}

    comp = CompositePostHocEvaluator(evaluators={"foo": Only()})
    _, metrics = comp.eval(
        [{"text": "y"}],
        dataloader_name="val/foo_prediction",
    )
    assert metrics == {"ok": 3.0}
