"""Validity-based metrics for maze path generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch
from torch import Tensor
from torchmetrics import MeanMetric

Grid = List[List[int]]


class MazeSuccess(MeanMetric):
    """Measure whether generated paths solve the fixed maze task.

    A path succeeds when every decoded cell is traversable, every transition
    is a four-neighbour move, and all conditioning subgoals occur in order.
    Inputs are SimpleSpaceTokenizer token IDs by default.
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        filename: str = "maze.json",
        *,
        grid: Optional[Sequence[Sequence[int]]] = None,
        metadata_path: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token_offset: int = 7,
        pad_token_id: int = 0,
        eos_token_id: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        source_count = sum(
            value is not None for value in (grid, metadata_path, repo_id)
        )
        if source_count != 1:
            raise ValueError(
                "provide exactly one of grid, metadata_path, or repo_id"
            )
        if token_offset < 0:
            raise ValueError("token_offset must be non-negative")

        if grid is not None:
            loaded_grid = [list(row) for row in grid]
        else:
            path = (
                Path(metadata_path)
                if metadata_path is not None
                else _download_metadata(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=cache_dir,
                )
            )
            with path.open() as file:
                metadata = json.load(file)
            if "grid" not in metadata:
                raise ValueError(f"maze metadata at {path} has no grid field")
            loaded_grid = metadata["grid"]

        self.grid = _validate_grid(loaded_grid)
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.token_offset = token_offset
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def update(
        self,
        pred_cells: Tensor,
        subgoal_cells: Tensor,
        pred_mask: Optional[Tensor] = None,
    ) -> None:
        """Add a batch of generated token sequences to the running mean."""
        predictions = _as_batch(pred_cells)
        subgoals = _as_batch(subgoal_cells)
        if predictions.shape[0] != subgoals.shape[0]:
            raise ValueError(
                "pred_cells and subgoal_cells must have equal batch sizes"
            )

        masks = None
        if pred_mask is not None:
            masks = _as_batch(pred_mask).to(dtype=torch.bool)
            if masks.shape != predictions.shape:
                raise ValueError(
                    "pred_mask must have the same shape as pred_cells"
                )

        success = []
        for index in range(predictions.shape[0]):
            path = self._decode_tokens(
                predictions[index],
                None if masks is None else masks[index],
            )
            anchors = self._decode_tokens(subgoals[index])
            success.append(
                path is not None
                and anchors is not None
                and self._is_valid_path(path, anchors)
            )

        values = torch.tensor(
            success,
            dtype=torch.float32,
            device=predictions.device,
        )
        super().update(values)
        self._computed_value = values

    def _decode_tokens(
        self,
        tokens: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Optional[List[int]]:
        cells = []
        token_values = tokens.detach().cpu().tolist()
        mask_values = (
            None if mask is None else mask.detach().cpu().tolist()
        )
        for index, token in enumerate(token_values):
            if mask_values is not None and not mask_values[index]:
                continue
            if token in (self.pad_token_id, self.eos_token_id):
                break
            cell = token - self.token_offset
            if cell < 0 or cell >= self.height * self.width:
                return None
            cells.append(cell)
        return cells

    def _is_valid_path(
        self,
        path: Sequence[int],
        subgoals: Sequence[int],
    ) -> bool:
        if not path or not subgoals:
            return False

        coordinates = [divmod(cell, self.width) for cell in path]
        if any(self.grid[row][col] != 0 for row, col in coordinates):
            return False
        if any(
            abs(left[0] - right[0]) + abs(left[1] - right[1]) != 1
            for left, right in zip(coordinates, coordinates[1:])
        ):
            return False

        next_subgoal = 0
        for cell in path:
            if cell == subgoals[next_subgoal]:
                next_subgoal += 1
                if next_subgoal == len(subgoals):
                    return True
        return False


def _as_batch(values: Tensor) -> Tensor:
    tensor = torch.as_tensor(values)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError("maze metric inputs must be rank-one or rank-two")
    return tensor


def _validate_grid(grid: Sequence[Sequence[int]]) -> Grid:
    if not grid or not grid[0]:
        raise ValueError("maze grid must be non-empty")
    width = len(grid[0])
    result = []
    for row in grid:
        if len(row) != width:
            raise ValueError("maze grid rows must have equal widths")
        values = [int(value) for value in row]
        if any(value not in (0, 1) for value in values):
            raise ValueError("maze grid values must be 0 or 1")
        result.append(values)
    return result


def _download_metadata(
    *,
    repo_id: Optional[str],
    filename: str,
    revision: Optional[str],
    cache_dir: Optional[str],
) -> Path:
    if repo_id is None:
        raise ValueError("repo_id is required for Hub metadata")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "loading maze metadata from the Hub requires huggingface_hub"
        ) from error

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
        )
    )
