"""Tests for maze generation, preprocessing, and validity metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pytest
import torch

from xlm.tasks.maze import on_the_fly_fn, preprocess_fn
from xlm.tasks.maze.generate import (
    Grid,
    free_cells,
    generate_examples,
    generate_maze,
    open_neighbors,
    shortest_path,
    unflatten_cell,
)
from xlm.tasks.maze.metrics import MazeSuccess

Cell = Tuple[int, int]


class _NumberTokenizer:
    def _convert_token_to_id(self, token: str) -> int:
        return int(token) + 7


def _edge_count(grid: Grid) -> int:
    degrees = (
        len(open_neighbors(grid, cell)) for cell in free_cells(grid)
    )
    return sum(degrees) // 2


def _cycle_rank(grid: Grid) -> int:
    return _edge_count(grid) - len(free_cells(grid)) + 1


def _reachable_cells(grid: Grid, start: Cell) -> set[Cell]:
    reached = {start}
    frontier = [start]
    while frontier:
        cell = frontier.pop()
        for neighbor in open_neighbors(grid, cell):
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    return reached


def _assert_valid_path(grid: Grid, path: List[Cell]) -> None:
    assert path
    assert all(grid[row][col] == 0 for row, col in path)
    assert all(
        abs(left[0] - right[0]) + abs(left[1] - right[1]) == 1
        for left, right in zip(path, path[1:])
    )


def test_maze_generation_is_deterministic() -> None:
    for family in ("perfect", "imperfect", "braided"):
        assert generate_maze(family, 10, seed=9) == generate_maze(
            family, 10, seed=9
        )


def test_perfect_maze_is_a_connected_tree() -> None:
    grid = generate_maze("perfect", 10, seed=2025)
    cells = free_cells(grid)
    assert len(grid) == 21
    assert all(len(row) == 21 for row in grid)
    assert _reachable_cells(grid, cells[0]) == set(cells)
    assert _edge_count(grid) == len(cells) - 1


def test_imperfect_and_braided_mazes_add_cycles() -> None:
    perfect = generate_maze("perfect", 10, seed=2025)
    imperfect = generate_maze("imperfect", 10, seed=2025)
    braided = generate_maze("braided", 10, seed=2025)

    assert _cycle_rank(perfect) == 0
    assert _cycle_rank(imperfect) > 0
    assert _cycle_rank(braided) > 0
    assert len(free_cells(imperfect)) > len(free_cells(perfect))
    assert len(free_cells(braided)) > len(free_cells(perfect))


def test_full_braiding_removes_dead_ends() -> None:
    braided = generate_maze(
        "braided",
        10,
        seed=2025,
        braid_fraction=1.0,
    )
    assert all(
        len(open_neighbors(braided, cell)) != 1
        for cell in free_cells(braided)
    )


def test_shortest_path_is_valid() -> None:
    grid = generate_maze("perfect", 10, seed=7)
    cells = free_cells(grid)
    path = shortest_path(grid, cells[0], cells[-1])
    _assert_valid_path(grid, path)
    assert path[0] == cells[0]
    assert path[-1] == cells[-1]


@pytest.mark.parametrize(
    ("family", "strategy"),
    [
        ("perfect", "subgoals_first"),
        ("imperfect", "path_first"),
        ("braided", "path_first"),
    ],
)
def test_generated_examples_have_valid_paths_and_anchors(
    family: str,
    strategy: str,
) -> None:
    grid = generate_maze(family, 10, seed=11)
    rows = generate_examples(
        grid,
        strategy=strategy,
        split_sizes={"train": 8, "validation": 3, "test": 3},
        min_subgoals=2,
        max_subgoals=4,
        max_path_length=128,
        paths_per_pair=3,
        seed=12,
    )

    endpoint_owner = {}
    seen_subgoals = set()
    width = len(grid[0])
    for split, examples in rows.items():
        for example in examples:
            path = [
                unflatten_cell(token, width) for token in example["path"]
            ]
            _assert_valid_path(grid, path)
            assert len(path) <= 128
            assert [
                example["path"][position]
                for position in example["subgoal_pos"]
            ] == example["subgoals"]

            endpoint = (
                example["subgoals"][0],
                example["subgoals"][-1],
            )
            assert endpoint_owner.setdefault(endpoint, split) == split
            subgoal_key = tuple(example["subgoals"])
            assert subgoal_key not in seen_subgoals
            seen_subgoals.add(subgoal_key)


def test_preprocess_and_on_the_fly_fields() -> None:
    tokenizer = _NumberTokenizer()
    row = {
        "path": [6, 7, 8, 13],
        "subgoals": [6, 8, 13],
        "subgoal_pos": [0, 2, 3],
    }
    processed = preprocess_fn(row, tokenizer)  # type: ignore[arg-type]
    assert processed["input_token_ids"] == [13, 14, 15, 20]
    assert processed["prompt_token_ids"] == [13, 15, 20]
    assert processed["anchor_pos"] == [0, 2, 3]

    model_row = on_the_fly_fn(
        processed,
        tokenizer,  # type: ignore[arg-type]
        block_size=4,
    )
    assert model_row == {
        "input_ids": [13, 14, 15, 20],
        "prompt_ids": [13, 15, 20],
        "anchor_pos": [0, 2, 3],
    }


def test_preprocess_rejects_incorrect_anchor_positions() -> None:
    row = {
        "path": [6, 7, 8],
        "subgoals": [6, 8],
        "subgoal_pos": [0, 1],
    }
    with pytest.raises(ValueError, match="does not point"):
        preprocess_fn(row, _NumberTokenizer())  # type: ignore[arg-type]


def test_on_the_fly_rejects_oversized_paths() -> None:
    row = {
        "input_token_ids": [13, 14, 15],
        "prompt_token_ids": [13, 15],
        "anchor_pos": [0, 2],
    }
    with pytest.raises(ValueError, match="exceeding block_size"):
        on_the_fly_fn(
            row,
            _NumberTokenizer(),  # type: ignore[arg-type]
            block_size=2,
        )


def test_maze_success_accepts_only_valid_conditioned_paths() -> None:
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    metric = MazeSuccess(grid=grid)
    metric.update(
        pred_cells=torch.tensor(
            [
                [13, 14, 15, 20, 25, 24, 23, 3],
                [13, 18, 3, 0, 0, 0, 0, 0],
                [13, 15, 20, 25, 24, 23, 3, 0],
                [13, 14, 15, 20, 25, 24, 23, 3],
                [13, 14, 15, 20, 25, 24, 23, 3],
            ]
        ),
        subgoal_cells=torch.tensor(
            [
                [13, 20, 23, 0],
                [13, 18, 0, 0],
                [13, 20, 23, 0],
                [13, 24, 20, 0],
                [13, 20, 22, 0],
            ]
        ),
    )
    assert metric.compute().item() == pytest.approx(0.2)


def test_maze_success_honors_prediction_mask() -> None:
    metric = MazeSuccess(grid=[[0, 0, 0]])
    metric.update(
        pred_cells=torch.tensor([[2, 7, 8, 9, 0]]),
        subgoal_cells=torch.tensor([[7, 9, 0]]),
        pred_mask=torch.tensor([[False, True, True, True, False]]),
    )
    assert metric.compute().item() == 1.0


def test_maze_success_loads_local_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "maze.json"
    metadata_path.write_text(json.dumps({"grid": [[0, 0]]}))
    metric = MazeSuccess(metadata_path=str(metadata_path))
    metric.update(
        pred_cells=torch.tensor([7, 8, 3]),
        subgoal_cells=torch.tensor([7, 8, 0]),
    )
    assert metric.compute().item() == 1.0
