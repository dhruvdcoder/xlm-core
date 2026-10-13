"""Preprocessing for brozonoyer/sudoku-extreme-multi-solution.

The modeling config (``default``) stores one puzzle and the full solution
set. Blanks are ``.``; grids are 81 characters. Load it as
``brozonoyer/sudoku-extreme-multi-solution/train`` (no config segment).
Do not point a training manager at the ``provenance`` or ``sweep``
configs: provenance records which clue was deleted, and sweep is an
upstream summary.

``solutions`` is an unordered set stored lexicographically. Preprocess
tokenizes every solution into ``solution_token_ids`` and does not
choose one. There is no on-the-fly selector: a custom collator decides
which member is the training target. ``solutions[0]`` is only the
sorted member, not a canonical target.

``single_deletion_filter_fn`` keeps the covariate-matched backbone
(N in {2, 3, 4, 6, 8, 16}). N=1 is ``redundant_control`` and is not
matched to that backbone. ``base_rating`` is a parent covariate and
does not describe the derived puzzle.
"""

from __future__ import annotations

from typing import Any, Dict, List

from xlm.datamodule import SimpleSpaceTokenizer

_GRID_LEN = 81


def _mask_blanks(
    token_ids: List[int], zero_id: int, mask_id: int
) -> List[int]:
    return [
        mask_id if token_id == zero_id else token_id
        for token_id in token_ids
    ]


def _tokenize_grid(
    grid: str,
    tokenizer: SimpleSpaceTokenizer,
    zero_id: int,
    mask_id: int,
    *,
    blanks: bool,
) -> List[int]:
    if len(grid) != _GRID_LEN:
        raise ValueError(
            f"expected an {_GRID_LEN}-character grid, got {len(grid)}"
        )
    if blanks:
        if any(ch not in ".123456789" for ch in grid):
            raise ValueError(f"puzzle has a non-grid character: {grid!r}")
        normalized = grid.replace(".", "0")
    else:
        if any(ch not in "123456789" for ch in grid):
            raise ValueError(
                f"solution has a blank or non-digit: {grid!r}"
            )
        normalized = grid
    token_ids = [
        tokenizer._convert_token_to_id(ch) for ch in normalized
    ]
    if blanks:
        return _mask_blanks(token_ids, zero_id, mask_id)
    return token_ids


def _solutions_agree(puzzle: str, solutions: List[str]) -> None:
    for solution in solutions:
        for puzzle_ch, solution_ch in zip(puzzle, solution):
            if puzzle_ch != "." and puzzle_ch != solution_ch:
                raise ValueError(
                    "solution disagrees with a given clue"
                )


def sudoku_multi_solution_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: SimpleSpaceTokenizer,
) -> Dict[str, Any]:
    """Tokenize the puzzle and every solution.

    Blanks become the mask id. The returned example has
    ``prompt_token_ids``, ``solution_token_ids``, and
    ``determined_token_ids``. It does not set ``input_token_ids``.
    """
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Mask token not found in tokenizer")
    zero_id = tokenizer._convert_token_to_id("0")
    puzzle = example["puzzle"]
    solutions = list(example["solutions"])
    if not solutions:
        raise ValueError("solutions is empty")
    if len(set(solutions)) != len(solutions):
        raise ValueError("solutions contains duplicates")
    num_solutions = example.get("num_solutions")
    if num_solutions is not None and int(num_solutions) != len(solutions):
        raise ValueError(
            f"num_solutions={num_solutions} but {len(solutions)} "
            "solutions were stored"
        )
    _solutions_agree(puzzle, solutions)
    example["prompt_token_ids"] = _tokenize_grid(
        puzzle, tokenizer, zero_id, mask_id, blanks=True
    )
    example["solution_token_ids"] = [
        _tokenize_grid(
            solution, tokenizer, zero_id, mask_id, blanks=False
        )
        for solution in solutions
    ]
    example["determined_token_ids"] = _tokenize_grid(
        example["determined_board"],
        tokenizer,
        zero_id,
        mask_id,
        blanks=True,
    )
    return example


def sudoku_multi_solution_single_deletion_filter_fn(
    example: Dict[str, Any],
) -> bool:
    """Keep puzzles made by deleting one clue from a unique parent.

    This is the covariate-matched subset for comparisons across N.
    It excludes ``redundant_control``, ``composed_n2``, and
    ``deletion_search``.
    """
    return example["construction"] == "single_deletion"


def sudoku_multi_solution_n1_filter_fn(example: Dict[str, Any]) -> bool:
    """Keep N=1 puzzles.

    These are ``redundant_control`` items and are not
    covariate-matched to the single-deletion backbone.
    """
    return example["num_solutions"] == 1


def sudoku_multi_solution_n2_filter_fn(example: Dict[str, Any]) -> bool:
    """Keep puzzles with ``num_solutions == 2``."""
    return example["num_solutions"] == 2


def sudoku_multi_solution_n3_filter_fn(example: Dict[str, Any]) -> bool:
    """Keep puzzles with ``num_solutions == 3``."""
    return example["num_solutions"] == 3


def sudoku_multi_solution_n4_filter_fn(example: Dict[str, Any]) -> bool:
    """Keep puzzles with ``num_solutions == 4``."""
    return example["num_solutions"] == 4


def sudoku_multi_solution_n6_filter_fn(example: Dict[str, Any]) -> bool:
    """Keep puzzles with ``num_solutions == 6``."""
    return example["num_solutions"] == 6


def sudoku_multi_solution_n8_filter_fn(example: Dict[str, Any]) -> bool:
    """Keep puzzles with ``num_solutions == 8``."""
    return example["num_solutions"] == 8


def sudoku_multi_solution_n16_filter_fn(example: Dict[str, Any]) -> bool:
    """Keep puzzles with ``num_solutions == 16``."""
    return example["num_solutions"] == 16
