# Maze planning

The maze task is a reconstruction of Appendix D.1 of
[Variational Learning for Insertion-based Generation](https://arxiv.org/abs/2606.02133),
which extends the task introduced by
[FlexMDM](https://arxiv.org/abs/2509.01025).

Each dataset repository contains one fixed maze. The maze is global task
state and is not included in each example. An example supplies ordered
subgoals, and the target is a path through free cells that visits those
subgoals in order.

The papers do not publish generation code, datasets, exact difficulty-tier
parameters, or a precise detour algorithm. The generator therefore separates
paper-specified settings from explicit reconstruction defaults.

## Maze families

- **Perfect:** recursive division creates a connected, acyclic grid graph.
- **Imperfect:** opens 30% of candidate walls separating existing passages.
- **Braided:** opens a wall at every initial dead end when the braid fraction
  is 1.0.

Cells use four-neighbor connectivity. A cell at row `r`, column `c` in a grid
of width `W` is represented by the integer `r * W + c`. Walls are `1` and
traversable cells are `0` in `maze.json`.

Perfect-maze examples use subgoals-first sampling: distinct cells are sampled
and consecutive subgoals are connected with BFS. Imperfect and braided
examples use path-first sampling: a shortest start-goal path is generated,
longer simple detours are attempted, and ordered interior cells become
subgoals.

## Reconstruction tiers

| Tier | Grid | Subgoals | Maximum path |
|------|------|----------|--------------|
| Easy | 21 by 21 (`m=10`) | 2–4 | 128 |
| Medium | 31 by 31 (`m=15`) | 3–8 | 256 |
| Hard | 41 by 41 (`m=20`) | 4–12 | 400 |

The hard grid size and 400-token maximum come from the Insertion Process
paper. The easy and medium settings, subgoal ranges, and dataset sizes are
reproduction choices. Default split sizes are 50,000 train, 5,000 validation,
and 5,000 test examples.

## Generate a dataset

From an editable xlm-core installation:

```bash
python -m xlm.tasks.maze.generate \
  --family perfect \
  --tier hard \
  --seed 2025 \
  --output-dir maze_data/perfect-hard
```

Add `--plot` to save a preview. To generate and upload in one command:

```bash
python -m xlm.tasks.maze.generate \
  --family perfect \
  --tier hard \
  --seed 2025 \
  --push brozonoyer/maze-perfect-hard
```

Hub upload requires an authenticated Hugging Face client. The dataset configs
target repositories named
`brozonoyer/maze-{perfect,imperfect,braided}-{easy,medium,hard}`.

The local output contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, and
`maze.json`. Each dataset row has:

| Field | Meaning |
|-------|---------|
| `path` | Flattened cell indices for the target trajectory |
| `subgoals` | Ordered conditioning cells |
| `subgoal_pos` | Exact positions of the subgoals in `path` |

`subgoal_pos` removes ambiguity when a subgoals-first trajectory revisits a
cell.

## xlm-core data fields

`xlm.tasks.maze.preprocess_fn` maps cell indices through
`SimpleSpaceTokenizer.for_numbers` and produces:

- `input_token_ids`: complete target path;
- `prompt_token_ids`: ordered subgoals;
- `anchor_pos`: target-path positions for anchored insertion models.

`xlm.tasks.maze.on_the_fly_fn` exposes these as `input_ids`, `prompt_ids`, and
`anchor_pos`. Keeping both prompt and anchor fields permits either a seq2seq
prefix formulation or an anchored-canvas formulation.

The shared dataset config is
`src/xlm/configs/lightning_train/datasets/maze.yaml`. Family/tier configs
follow the pattern `maze_<family>_<tier>_<split>.yaml`, including separate
validation/test prediction variants.

## Maze success

`xlm.tasks.maze.metrics.MazeSuccess` reports validity rather than strict target
path equality. A generated sequence succeeds when:

1. every decoded cell is traversable;
2. every consecutive pair is a four-neighbor move; and
3. all conditioning subgoals occur in order as a subsequence.

The metric can load `maze.json` from its dataset repository or from a local
path. The reusable Hydra wrapper is
`src/xlm/configs/lightning_train/metrics/maze_success.yaml`; model-specific
configuration supplies its repository, prefix, and update function.
