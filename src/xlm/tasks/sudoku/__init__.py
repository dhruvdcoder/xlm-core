from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from pathlib import Path
from xlm.datamodule import DatasetManager
import datasets

from xlm.datamodule import SimpleSpaceTokenizer
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


def _replace(tokens: List[int], zero_id: int, mask_id: int) -> List[int]:
    return [mask_id if c == zero_id else c for c in tokens]


def sudoku_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: SimpleSpaceTokenizer,
) -> Dict[str, Any]:
    partial_sequence: str = example["quizzes"]
    ground_truth_sequence: str = example["solutions"]
    prompt_ids: List[int] = [
        tokenizer._convert_token_to_id(str(ch)) for ch in partial_sequence
    ]
    # "0" in partial_sequence represents the blank positions
    zero_id = tokenizer._convert_token_to_id("0")
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Mask token not found in tokenizer")
    prompt_ids = _replace(prompt_ids, zero_id, mask_id)
    input_ids = [
        tokenizer._convert_token_to_id(str(ch)) for ch in ground_truth_sequence
    ]
    input_ids = _replace(input_ids, zero_id, mask_id)
    example["input_token_ids"] = input_ids
    example["prompt_token_ids"] = prompt_ids
    return example


class LocalDatasetManager(DatasetManager):
    def __init__(
        self,
        *args,
        ds_type: Optional[str] = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if ds_type is None:
            raise ValueError("ds_type is required")
        super().__init__(*args, **kwargs)
        self.load_kwargs = load_kwargs or {}
        self.ds_type = ds_type

    def _download(self, num_proc: Optional[int] = None) -> datasets.Dataset:
        if self.ds_type == "csv":
            file_name = f"{self._split_to_download}.csv"
            _path = Path(self.full_name).parent
            data_files = str(_path / file_name)
            ds = datasets.load_dataset(
                "csv",
                data_files=data_files,
                **self.load_kwargs,
                num_proc=num_proc,
            )["train"]
            return ds
        else:
            raise ValueError(f"Unsupported dataset type: {self.ds_type}")
