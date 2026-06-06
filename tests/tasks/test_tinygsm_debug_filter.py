"""Tests for TinyGSM debug one-example filter."""

import datasets

from xlm.tasks.tinygsm import (
    reset_tinygsm_debug_first_example_filter_fn,
    tinygsm_debug_first_example_filter_fn,
)


def test_tinygsm_debug_first_example_filter_fn_keeps_one_row() -> None:
    reset_tinygsm_debug_first_example_filter_fn()
    ds = datasets.Dataset.from_dict(
        {
            "question": ["q0", "q1", "q2"],
            "code": ["c0", "c1", "c2"],
        }
    )
    filtered = ds.filter(tinygsm_debug_first_example_filter_fn)
    assert len(filtered) == 1
    assert filtered[0]["question"] == "q0"
    assert filtered[0]["code"] == "c0"


def test_tinygsm_debug_first_example_filter_fn_reset() -> None:
    reset_tinygsm_debug_first_example_filter_fn()
    ds = datasets.Dataset.from_dict({"question": ["a"], "code": ["b"]})
    assert len(ds.filter(tinygsm_debug_first_example_filter_fn)) == 1
    reset_tinygsm_debug_first_example_filter_fn()
    ds2 = datasets.Dataset.from_dict({"question": ["x"], "code": ["y"]})
    assert len(ds2.filter(tinygsm_debug_first_example_filter_fn)) == 1
    assert ds2[0]["question"] == "x"
