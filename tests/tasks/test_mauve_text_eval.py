"""Unit tests for ``MauveTextEval`` with mocked ``mauve`` / HF streaming.

Root ``tests/conftest.py`` imports ``transformers`` via ``xlm.datamodule``. For a
minimal collect/run without those deps (or when the env breaks ``transformers``
imports), use::

    pytest tests/tasks/test_mauve_text_eval.py --confcutdir=tests/tasks -o addopts=
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from xlm.tasks.owt.mauve_text_eval import MauveTextEval


def _fake_mauve_out(mauve: float = 0.42, fi: float = 0.11, star: bool = False):
    ns = SimpleNamespace(mauve=mauve, frontier_integral=fi)
    if star:
        ns.mauve_star = 0.99
        ns.frontier_integral_star = 0.88
    return ns


def test_eval_per_row_triggers_compute_mauve_and_aggregates():
    preds = [
        {"text": " gen one ", "truth": "human reference one is long enough"},
        {"text": "gen two", "truth": "human reference two is long enough"},
    ]
    fake_mauve_mod = MagicMock()
    fake_mauve_mod.compute_mauve = MagicMock(
        return_value=_fake_mauve_out(star=True)
    )
    ev = MauveTextEval()
    with patch.dict(sys.modules, {"mauve": fake_mauve_mod}):
        out, agg = ev.eval(preds, tokenizer=None, dataloader_name="lm_prediction")

    fake_mauve_mod.compute_mauve.assert_called_once()
    assert agg["mauve"] == 0.42
    assert agg["mauve_frontier_integral"] == 0.11
    assert agg["mauve_num_pairs"] == 2
    assert agg["mauve_star"] == 0.99
    assert agg["mauve_frontier_integral_star"] == 0.88

    assert out[0]["mauve_included"] is True
    assert out[0]["mauve_score"] == 0.42
    assert "human reference one"[:200] in out[0]["mauve_reference_excerpt"]
    assert "mauve_eval_meta" in out[0]


def test_eval_hf_streaming_uses_mocked_load_dataset():
    preds = [
        {"text": "machine output number one is here"},
        {"text": "machine output number two is here"},
    ]

    class _FakeShuffled:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

    class _FakeIterDs:
        def __init__(self, rows):
            self._rows = rows

        def shuffle(self, seed, buffer_size):
            return _FakeShuffled(self._rows)

    def _fake_load_dataset(path, split, streaming=True):
        del path, split, streaming
        return _FakeIterDs(
            [
                {"text": "human streamed text one has enough chars"},
                {"text": "human streamed text two has enough chars"},
            ]
        )

    fake_mauve_mod = MagicMock()
    fake_mauve_mod.compute_mauve = MagicMock(return_value=_fake_mauve_out())

    ev = MauveTextEval(human_text_source="hf_streaming")
    with patch.dict(sys.modules, {"mauve": fake_mauve_mod}):
        with patch(
            "datasets.load_dataset", side_effect=_fake_load_dataset
        ) as ld:
            out, agg = ev.eval(preds, tokenizer=None)

    ld.assert_called_once()
    fake_mauve_mod.compute_mauve.assert_called_once()
    assert agg["mauve"] == 0.42
    assert agg["mauve_num_pairs"] == 2

    assert out[0]["mauve_included"] is True
    assert out[0]["mauve_reference_source"] == "hf_streaming"
    assert out[0]["mauve_score"] == 0.42
    meta = out[0]["mauve_eval_meta"]
    assert meta["mauve_human_text_source"] == "hf_streaming"
