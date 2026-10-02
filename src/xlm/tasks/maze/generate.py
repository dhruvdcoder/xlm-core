"""Generate the maze-planning datasets from the Insertion Process paper.

The implementation follows Appendix D.1 of "Variational Learning for
Insertion-based Generation" and the recursive-division listing in FlexMDM.
The paper does not publish its generator or exact easy/medium/hard settings,
so the tier defaults in this module are explicit, reproducible choices.

Each dataset repository contains one fixed maze. Examples contain:

* path: a valid trajectory represented by flattened cell indices;
* subgoals: an ordered subset of cells that the path visits; and
* subgoal_pos: the corresponding positions in path.

Run python -m xlm.tasks.maze.generate --help for the CLI.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

Cell = Tuple[int, int]
Grid = List[List[int]]
MazeFamily = Literal["perfect", "imperfect", "braided"]
SamplingStrategy = Literal["subgoals_first", "path_first"]

CARDINAL_DIRECTIONS: Tuple[Cell, ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


@dataclass(frozen=True)
class TierConfig:
    """Published and inferred difficulty controls for one dataset tier."""

    maze_size: int
    min_subgoals: int
    max_subgoals: int
    max_path_length: int


TIER_CONFIGS: Dict[str, TierConfig] = {
    "easy": TierConfig(
        maze_size=10,
        min_subgoals=2,
        max_subgoals=4,
        max_path_length=128,
    ),
    "medium": TierConfig(
        maze_size=15,
        min_subgoals=3,
        max_subgoals=8,
        max_path_length=256,
    ),
    "hard": TierConfig(
        maze_size=20,
        min_subgoals=4,
        max_subgoals=12,
        max_path_length=400,
    ),
}


def _divide(
    grid: Grid,
    top: int,
    left: int,
    height: int,
    width: int,
    rng: random.Random,
) -> None:
    """Recursively divide an open rectangle with one doorway per new wall."""
    if height <= 2 or width <= 2:
        return

    horizontal = width < height
    if horizontal:
        wall_row = rng.randrange(top + 1, top + height - 1, 2)
        gap_col = rng.randrange(left, left + width, 2)
        grid[wall_row][left : left + width] = [1] * width
        grid[wall_row][gap_col] = 0
        _divide(grid, top, left, wall_row - top, width, rng)
        _divide(
            grid,
            wall_row + 1,
            left,
            top + height - wall_row - 1,
            width,
            rng,
        )
    else:
        wall_col = rng.randrange(left + 1, left + width - 1, 2)
        gap_row = rng.randrange(top, top + height, 2)
        for row in range(top, top + height):
            grid[row][wall_col] = 1
        grid[gap_row][wall_col] = 0
        _divide(grid, top, left, height, wall_col - left, rng)
        _divide(
            grid,
            top,
            wall_col + 1,
            height,
            left + width - wall_col - 1,
            rng,
        )


def perfect_maze(maze_size: int, seed: int = 2025) -> Grid:
    """Generate the recursive-division base maze on a (2m+1)-square grid."""
    if maze_size < 2:
        raise ValueError(f"maze_size must be at least 2, got {maze_size}")

    side = 2 * maze_size + 1
    grid = [[0] * side for _ in range(side)]
    grid[0] = [1] * side
    grid[-1] = [1] * side
    for row in grid:
        row[0] = 1
        row[-1] = 1

    _divide(
        grid,
        top=1,
        left=1,
        height=side - 2,
        width=side - 2,
        rng=random.Random(seed),
    )
    return grid


def imperfect_maze(
    grid: Grid,
    door_fraction: float = 0.3,
    seed: int = 2025,
) -> Grid:
    """Open a fraction of walls separating two passages to create cycles."""
    _validate_fraction("door_fraction", door_fraction)
    result = _copy_grid(grid)
    height, width = _grid_shape(result)
    candidates: List[Cell] = []
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if result[row][col] != 1:
                continue
            separates_ns = (
                result[row - 1][col] == 0
                and result[row + 1][col] == 0
            )
            separates_ew = (
                result[row][col - 1] == 0
                and result[row][col + 1] == 0
            )
            if separates_ns or separates_ew:
                candidates.append((row, col))

    rng = random.Random(seed)
    count = math.floor(door_fraction * len(candidates))
    for row, col in rng.sample(candidates, count):
        result[row][col] = 0
    return result


def braided_maze(
    grid: Grid,
    braid_fraction: float = 1.0,
    seed: int = 2025,
) -> Grid:
    """Remove walls at a fraction of dead ends to connect existing passages."""
    _validate_fraction("braid_fraction", braid_fraction)
    result = _copy_grid(grid)
    dead_ends = [
        cell
        for cell in free_cells(result)
        if len(open_neighbors(result, cell)) == 1
    ]
    rng = random.Random(seed)
    rng.shuffle(dead_ends)
    selected = dead_ends[: math.floor(braid_fraction * len(dead_ends))]

    height, width = _grid_shape(result)
    for row, col in selected:
        candidates: List[Cell] = []
        for delta_row, delta_col in CARDINAL_DIRECTIONS:
            wall = (row + delta_row, col + delta_col)
            passage = (row + 2 * delta_row, col + 2 * delta_col)
            if not (
                0 <= passage[0] < height and 0 <= passage[1] < width
            ):
                continue
            if (
                result[wall[0]][wall[1]] == 1
                and result[passage[0]][passage[1]] == 0
            ):
                candidates.append(wall)
        if candidates:
            wall_row, wall_col = rng.choice(candidates)
            result[wall_row][wall_col] = 0
    return result


def generate_maze(
    family: MazeFamily,
    maze_size: int,
    seed: int = 2025,
    door_fraction: float = 0.3,
    braid_fraction: float = 1.0,
) -> Grid:
    """Generate one fixed maze from the requested family."""
    base = perfect_maze(maze_size, seed)
    if family == "perfect":
        return base
    if family == "imperfect":
        return imperfect_maze(base, door_fraction, seed + 1)
    if family == "braided":
        return braided_maze(base, braid_fraction, seed + 1)
    raise ValueError(f"unknown maze family: {family!r}")


def open_neighbors(grid: Grid, cell: Cell) -> List[Cell]:
    """Return traversable 4-neighbours of cell."""
    height, width = _grid_shape(grid)
    row, col = cell
    neighbors = []
    for delta_row, delta_col in CARDINAL_DIRECTIONS:
        other_row, other_col = row + delta_row, col + delta_col
        if (
            0 <= other_row < height
            and 0 <= other_col < width
            and grid[other_row][other_col] == 0
        ):
            neighbors.append((other_row, other_col))
    return neighbors


def free_cells(grid: Grid) -> List[Cell]:
    """List all traversable cells in row-major order."""
    return [
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value == 0
    ]


def shortest_path(
    grid: Grid,
    start: Cell,
    goal: Cell,
    *,
    forbidden: Optional[Iterable[Cell]] = None,
    rng: Optional[random.Random] = None,
) -> List[Cell]:
    """Find a BFS shortest path, or return an empty list if none exists."""
    height, width = _grid_shape(grid)
    for name, cell in (("start", start), ("goal", goal)):
        row, col = cell
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{name} cell {cell} is outside the grid")
        if grid[row][col] != 0:
            raise ValueError(f"{name} cell {cell} is not traversable")

    blocked = set(forbidden or ())
    blocked.discard(start)
    blocked.discard(goal)
    queue = deque([start])
    parent: Dict[Cell, Optional[Cell]] = {start: None}

    while queue:
        cell = queue.popleft()
        if cell == goal:
            return _reconstruct_path(parent, goal)
        neighbors = open_neighbors(grid, cell)
        if rng is not None:
            rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor in blocked or neighbor in parent:
                continue
            parent[neighbor] = cell
            queue.append(neighbor)
    return []


def connect_subgoals(
    grid: Grid, subgoals: Sequence[Cell]
) -> Tuple[List[Cell], List[int]]:
    """Join subgoals with BFS segments and return path/anchor positions."""
    if len(subgoals) < 2:
        raise ValueError("at least two subgoals are required")

    path = [subgoals[0]]
    positions = [0]
    for start, goal in zip(subgoals, subgoals[1:]):
        segment = shortest_path(grid, start, goal)
        if not segment:
            return [], []
        path.extend(segment[1:])
        positions.append(len(path) - 1)
    return path, positions


def alternative_simple_paths(
    grid: Grid,
    shortest: Sequence[Cell],
    *,
    count: int,
    rng: random.Random,
    max_attempts: int = 500,
) -> List[List[Cell]]:
    """Create longer paths by replacing a shortest-path segment with a detour.

    The replacement cannot touch the remainder of the original path, except
    at the selected segment endpoints. This is the path-first construction
    described in Appendix D.1 of the Insertion Process paper.
    """
    if count <= 0 or len(shortest) < 3:
        return []

    original = list(shortest)
    seen = {tuple(original)}
    alternatives: List[List[Cell]] = []
    for _ in range(max_attempts):
        if len(alternatives) >= count:
            break
        left = rng.randrange(0, len(original) - 2)
        right = rng.randrange(left + 2, len(original))
        blocked = set(original)
        blocked.discard(original[left])
        blocked.discard(original[right])
        detour = shortest_path(
            grid,
            original[left],
            original[right],
            forbidden=blocked,
            rng=rng,
        )
        if not detour or len(detour) <= right - left + 1:
            continue
        candidate = original[:left] + detour + original[right + 1 :]
        key = tuple(candidate)
        if len(set(candidate)) != len(candidate) or key in seen:
            continue
        seen.add(key)
        alternatives.append(candidate)
    return alternatives


def subgoals_from_path(
    path: Sequence[Cell],
    count: int,
    rng: random.Random,
) -> Tuple[List[Cell], List[int]]:
    """Select path endpoints and uniformly sampled interior positions."""
    if count < 2:
        raise ValueError(f"subgoal count must be at least 2, got {count}")
    if count > len(path):
        raise ValueError(
            f"cannot select {count} subgoals from path of length {len(path)}"
        )

    positions = [0]
    if count > 2:
        positions.extend(
            sorted(rng.sample(range(1, len(path) - 1), count - 2))
        )
    positions.append(len(path) - 1)
    return [path[position] for position in positions], positions


def flatten_cell(cell: Cell, width: int) -> int:
    """Map (row, col) to the token used by the benchmark."""
    return cell[0] * width + cell[1]


def unflatten_cell(token: int, width: int) -> Cell:
    """Invert flatten_cell."""
    return divmod(token, width)


def generate_examples(
    grid: Grid,
    *,
    strategy: SamplingStrategy,
    split_sizes: Dict[str, int],
    min_subgoals: int,
    max_subgoals: int,
    max_path_length: int,
    paths_per_pair: int = 10,
    seed: int = 2025,
    max_attempts_per_example: int = 1000,
) -> Dict[str, List[Dict[str, List[int]]]]:
    """Generate exact split sizes with endpoint pairs kept split-disjoint."""
    if min_subgoals < 2 or max_subgoals < min_subgoals:
        raise ValueError(
            "subgoal bounds must satisfy 2 <= min_subgoals <= max_subgoals"
        )
    if max_path_length < 2:
        raise ValueError("max_path_length must be at least 2")
    if paths_per_pair < 1:
        raise ValueError("paths_per_pair must be at least 1")
    if any(size < 0 for size in split_sizes.values()):
        raise ValueError("split sizes must be non-negative")

    rng = random.Random(seed)
    cells = free_cells(grid)
    if len(cells) < max_subgoals:
        raise ValueError(
            f"maze has {len(cells)} free cells, fewer than max_subgoals="
            f"{max_subgoals}"
        )

    _, width = _grid_shape(grid)
    examples = {split: [] for split in split_sizes}
    endpoint_owner: Dict[Tuple[Cell, Cell], str] = {}
    seen_subgoals: set[Tuple[Cell, ...]] = set()

    for split, target_size in split_sizes.items():
        attempts = 0
        max_attempts = max(max_attempts_per_example * target_size, 1)
        while len(examples[split]) < target_size:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"generated only {len(examples[split])}/{target_size} "
                    f"examples for {split!r} after {max_attempts} attempts; "
                    "relax the generation parameters"
                )

            candidates = _sample_candidate_group(
                grid,
                cells,
                strategy=strategy,
                min_subgoals=min_subgoals,
                max_subgoals=max_subgoals,
                max_path_length=max_path_length,
                paths_per_pair=paths_per_pair,
                rng=rng,
            )
            for path, subgoals, positions in candidates:
                endpoint_key = (subgoals[0], subgoals[-1])
                owner = endpoint_owner.get(endpoint_key)
                if owner is not None and owner != split:
                    continue
                subgoal_key = tuple(subgoals)
                if subgoal_key in seen_subgoals:
                    continue

                endpoint_owner[endpoint_key] = split
                seen_subgoals.add(subgoal_key)
                examples[split].append(
                    {
                        "path": [
                            flatten_cell(cell, width) for cell in path
                        ],
                        "subgoals": [
                            flatten_cell(cell, width) for cell in subgoals
                        ],
                        "subgoal_pos": positions,
                    }
                )
                if len(examples[split]) >= target_size:
                    break
    return examples


def write_dataset(
    output_dir: Path,
    examples: Dict[str, List[Dict[str, List[int]]]],
    metadata: Dict[str, object],
) -> None:
    """Write JSONL splits and maze metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in examples.items():
        with (output_dir / f"{split}.jsonl").open("w") as file:
            for row in rows:
                file.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (output_dir / "maze.json").open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")


def push_dataset(
    repo_id: str,
    examples: Dict[str, List[Dict[str, List[int]]]],
    metadata_path: Path,
    *,
    private: bool = False,
) -> None:
    """Push split rows and maze.json to one Hugging Face dataset repository."""
    try:
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Hub upload requires the datasets and huggingface_hub packages"
        ) from error

    dataset = DatasetDict(
        {
            split: Dataset.from_list(rows)
            for split, rows in examples.items()
        }
    )
    dataset.push_to_hub(repo_id, private=private)
    HfApi().upload_file(
        path_or_fileobj=str(metadata_path),
        path_in_repo="maze.json",
        repo_id=repo_id,
        repo_type="dataset",
    )


def plot_maze(
    grid: Grid,
    output_path: Path,
    *,
    path: Optional[Sequence[Cell]] = None,
    subgoals: Optional[Sequence[Cell]] = None,
) -> None:
    """Save a maze image, optionally overlaying one path and its subgoals."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("plotting requires matplotlib") from error

    figure, axis = plt.subplots()
    axis.imshow(grid, cmap="binary")
    if path:
        axis.plot(
            [cell[1] for cell in path],
            [cell[0] for cell in path],
            color="tab:red",
            linewidth=1,
        )
    if subgoals:
        axis.scatter(
            [cell[1] for cell in subgoals],
            [cell[0] for cell in subgoals],
            color="tab:blue",
            s=16,
        )
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=("perfect", "imperfect", "braided"),
        required=True,
    )
    parser.add_argument(
        "--tier",
        choices=tuple(TIER_CONFIGS),
        default="hard",
    )
    parser.add_argument(
        "--strategy",
        choices=("subgoals_first", "path_first"),
        default=None,
        help="Defaults to subgoals_first for perfect, path_first otherwise.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--maze-size", type=int)
    parser.add_argument("--min-subgoals", type=int)
    parser.add_argument("--max-subgoals", type=int)
    parser.add_argument("--max-path-length", type=int)
    parser.add_argument("--door-fraction", type=float, default=0.3)
    parser.add_argument("--braid-fraction", type=float, default=1.0)
    parser.add_argument("--paths-per-pair", type=int, default=10)
    parser.add_argument("--num-train", type=int, default=50_000)
    parser.add_argument("--num-validation", type=int, default=5_000)
    parser.add_argument("--num-test", type=int, default=5_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to maze_data/<family>-<tier>.",
    )
    parser.add_argument(
        "--push",
        metavar="NAMESPACE/REPOSITORY",
        help="Also push the generated splits and maze.json to the Hub.",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--plot",
        nargs="?",
        const="maze.png",
        help="Save a maze preview, optionally to the supplied filename.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    tier = TIER_CONFIGS[args.tier]
    maze_size = (
        tier.maze_size if args.maze_size is None else args.maze_size
    )
    min_subgoals = (
        tier.min_subgoals
        if args.min_subgoals is None
        else args.min_subgoals
    )
    max_subgoals = (
        tier.max_subgoals
        if args.max_subgoals is None
        else args.max_subgoals
    )
    max_path_length = (
        tier.max_path_length
        if args.max_path_length is None
        else args.max_path_length
    )
    strategy: SamplingStrategy = args.strategy or (
        "subgoals_first" if args.family == "perfect" else "path_first"
    )
    output_dir = args.output_dir or Path(
        "maze_data", f"{args.family}-{args.tier}"
    )

    grid = generate_maze(
        args.family,
        maze_size,
        seed=args.seed,
        door_fraction=args.door_fraction,
        braid_fraction=args.braid_fraction,
    )
    split_sizes = {
        "train": args.num_train,
        "validation": args.num_validation,
        "test": args.num_test,
    }
    examples = generate_examples(
        grid,
        strategy=strategy,
        split_sizes=split_sizes,
        min_subgoals=min_subgoals,
        max_subgoals=max_subgoals,
        max_path_length=max_path_length,
        paths_per_pair=args.paths_per_pair,
        seed=args.seed + 2,
    )
    height, width = _grid_shape(grid)
    metadata: Dict[str, object] = {
        "family": args.family,
        "tier": args.tier,
        "seed": args.seed,
        "grid": grid,
        "height": height,
        "width": width,
        "vocab_size": height * width,
        "strategy": strategy,
        "door_fraction": args.door_fraction,
        "braid_fraction": args.braid_fraction,
        "paths_per_pair": args.paths_per_pair,
        "split_sizes": split_sizes,
        "tier_config": asdict(
            TierConfig(
                maze_size=maze_size,
                min_subgoals=min_subgoals,
                max_subgoals=max_subgoals,
                max_path_length=max_path_length,
            )
        ),
        "source": {
            "paper": "Variational Learning for Insertion-based Generation",
            "arxiv": "2606.02133",
            "appendix": "D.1",
            "note": (
                "The paper does not publish code or exact tier parameters; "
                "see tier_config for this reproduction."
            ),
        },
    }
    write_dataset(output_dir, examples, metadata)

    if args.plot:
        first = next(
            (
                row
                for rows in examples.values()
                for row in rows
            ),
            None,
        )
        preview_path = (
            [unflatten_cell(token, width) for token in first["path"]]
            if first
            else None
        )
        preview_subgoals = (
            [unflatten_cell(token, width) for token in first["subgoals"]]
            if first
            else None
        )
        plot_maze(
            grid,
            output_dir / args.plot,
            path=preview_path,
            subgoals=preview_subgoals,
        )

    if args.push:
        push_dataset(
            args.push,
            examples,
            output_dir / "maze.json",
            private=args.private,
        )

    counts = ", ".join(
        f"{split}={len(rows)}" for split, rows in examples.items()
    )
    print(f"Wrote {args.family}/{args.tier} maze dataset to {output_dir}")
    print(counts)


def _sample_candidate_group(
    grid: Grid,
    cells: Sequence[Cell],
    *,
    strategy: SamplingStrategy,
    min_subgoals: int,
    max_subgoals: int,
    max_path_length: int,
    paths_per_pair: int,
    rng: random.Random,
) -> List[Tuple[List[Cell], List[Cell], List[int]]]:
    if strategy == "subgoals_first":
        subgoal_count = rng.randint(min_subgoals, max_subgoals)
        subgoals = rng.sample(list(cells), subgoal_count)
        path, positions = connect_subgoals(grid, subgoals)
        if not path or len(path) > max_path_length:
            return []
        return [(path, subgoals, positions)]

    if strategy != "path_first":
        raise ValueError(f"unknown sampling strategy: {strategy!r}")

    start, goal = rng.sample(list(cells), 2)
    base_path = shortest_path(grid, start, goal, rng=rng)
    if len(base_path) < 2:
        return []
    paths = [base_path]
    paths.extend(
        alternative_simple_paths(
            grid,
            base_path,
            count=paths_per_pair - 1,
            rng=rng,
        )
    )
    group = []
    for path in paths:
        if len(path) > max_path_length:
            continue
        maximum = min(max_subgoals, len(path))
        if maximum < min_subgoals:
            continue
        subgoal_count = rng.randint(min_subgoals, maximum)
        subgoals, positions = subgoals_from_path(path, subgoal_count, rng)
        group.append((path, subgoals, positions))
    return group


def _copy_grid(grid: Grid) -> Grid:
    return [row.copy() for row in grid]


def _grid_shape(grid: Grid) -> Tuple[int, int]:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid rows must have equal widths")
    return len(grid), width


def _reconstruct_path(
    parent: Dict[Cell, Optional[Cell]], goal: Cell
) -> List[Cell]:
    path = []
    current: Optional[Cell] = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


if __name__ == "__main__":
    main()
