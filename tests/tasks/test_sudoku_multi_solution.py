"""Tests for the multi-solution Sudoku preprocess."""

import pytest

from xlm.datamodule import SimpleSpaceTokenizer
from xlm.tasks.sudoku_multi_solution import (
    sudoku_multi_solution_n1_filter_fn,
    sudoku_multi_solution_n2_filter_fn,
    sudoku_multi_solution_n16_filter_fn,
    sudoku_multi_solution_preprocess_fn,
    sudoku_multi_solution_single_deletion_filter_fn,
)


def _tokenizer() -> SimpleSpaceTokenizer:
    return SimpleSpaceTokenizer.for_numbers(10)


def _example() -> dict:
    # Two completed grids that differ everywhere, plus one given.
    puzzle = "5" + ("." * 80)
    solution_a = "5" + ("1" * 80)
    solution_b = "5" + ("2" * 80)
    return {
        "puzzle": puzzle,
        "solutions": [solution_a, solution_b],
        "num_solutions": 2,
        "determined_board": "5" + ("." * 80),
        "construction": "single_deletion",
    }


def test_preprocess_keeps_every_solution_and_masks_blanks() -> None:
    tokenizer = _tokenizer()
    processed = sudoku_multi_solution_preprocess_fn(_example(), tokenizer)
    mask_id = tokenizer.mask_token_id
    digit_5 = tokenizer._convert_token_to_id("5")
    digit_1 = tokenizer._convert_token_to_id("1")
    digit_2 = tokenizer._convert_token_to_id("2")

    assert "input_token_ids" not in processed
    assert "input_ids" not in processed
    assert len(processed["solution_token_ids"]) == 2
    assert processed["prompt_token_ids"][0] == digit_5
    assert processed["prompt_token_ids"][1:] == [mask_id] * 80
    assert processed["solution_token_ids"] == [
        [digit_5] + [digit_1] * 80,
        [digit_5] + [digit_2] * 80,
    ]
    assert processed["determined_token_ids"][0] == digit_5
    assert processed["determined_token_ids"][1:] == [mask_id] * 80


def test_preprocess_rejects_a_solution_that_breaks_a_given() -> None:
    tokenizer = _tokenizer()
    example = _example()
    example["solutions"] = ["6" + ("1" * 80), "5" + ("2" * 80)]
    with pytest.raises(ValueError, match="disagrees"):
        sudoku_multi_solution_preprocess_fn(example, tokenizer)


def test_preprocess_rejects_a_count_mismatch() -> None:
    tokenizer = _tokenizer()
    example = _example()
    example["num_solutions"] = 3
    with pytest.raises(ValueError, match="num_solutions"):
        sudoku_multi_solution_preprocess_fn(example, tokenizer)


def test_filters() -> None:
    single = {"construction": "single_deletion", "num_solutions": 2}
    control = {"construction": "redundant_control", "num_solutions": 1}
    composed = {"construction": "composed_n2", "num_solutions": 16}
    assert sudoku_multi_solution_single_deletion_filter_fn(single)
    assert not sudoku_multi_solution_single_deletion_filter_fn(control)
    assert not sudoku_multi_solution_single_deletion_filter_fn(composed)
    assert sudoku_multi_solution_n1_filter_fn(control)
    assert sudoku_multi_solution_n2_filter_fn(single)
    assert sudoku_multi_solution_n16_filter_fn(composed)
    assert not sudoku_multi_solution_n16_filter_fn(single)
