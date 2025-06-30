import csv
import json
from collections import OrderedDict
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


from .rank_zero import rank_zero_only

log = logging.getLogger(__name__) 

@rank_zero_only
def mkdir_rank_zero_only(dir: Path, exist_ok: bool = True) -> None:
    """Create directory only on rank 0.

    Args:
        dir (Path): Directory path.
        exist_ok (bool, optional): If True, do not raise an exception if the
            directory already exists. Default to True.
    """

    dir.mkdir(parents=True, exist_ok=exist_ok)


def process_state_dict(
    state_dict: Union[OrderedDict, dict],
    symbols: int = 0,
    exceptions: Optional[Union[str, List[str]]] = None,
) -> OrderedDict:
    """Filter and map model state dict keys.

    Args:
        state_dict (Union[OrderedDict, dict]): State dict.
        symbols (int): Determines how many symbols should be cut in the
            beginning of state dict keys. Default to 0.
        exceptions (Union[str, List[str]], optional): Determines exceptions,
            i.e. substrings, which keys should not contain.

    Returns:
        OrderedDict: Filtered state dict.
    """

    new_state_dict = OrderedDict()
    if exceptions:
        if isinstance(exceptions, str):
            exceptions = [exceptions]
    for key, value in state_dict.items():
        is_exception = False
        if exceptions:
            for exception in exceptions:
                if key.startswith(exception):
                    is_exception = True
        if not is_exception:
            new_state_dict[key[symbols:]] = value

    return new_state_dict


def save_predictions_from_dataloader(predictions: List[Any], path: Path) -> None:
    """Save predictions returned by `Trainer.predict` method for single
    dataloader.

    Args:
        predictions (List[Any]): Predictions returned by `Trainer.predict` method.
        path (Path): Path to predictions.
    """

    if path.suffix == ".csv":
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file)
            for batch in predictions:
                keys = list(batch.keys())
                batch_size = len(batch[keys[0]])
                for i in range(batch_size):
                    row = {key: batch[key][i].tolist() for key in keys}
                    writer.writerow(row)

    elif path.suffix == ".json":
        processed_predictions = {}
        for batch in predictions:
            keys = [key for key in batch.keys() if key != "names"]
            batch_size = len(batch[keys[0]])
            for i in range(batch_size):
                item = {key: batch[key][i].tolist() for key in keys}
                if "names" in batch.keys():
                    processed_predictions[batch["names"][i]] = item
                else:
                    processed_predictions[len(processed_predictions)] = item
        with open(path, "w") as json_file:
            json.dump(processed_predictions, json_file, ensure_ascii=False)

    else:
        raise NotImplementedError(f"{path.suffix} is not implemented!")


def save_predictions(
    predictions: List[Any], dirname: str, output_format: str = "json"
) -> None:
    """Save predictions returned by `Trainer.predict` method.

    Due to `LightningDataModule.predict_dataloader` return type is
    Union[DataLoader, List[DataLoader]], so `Trainer.predict` method can return
    a list of dictionaries, one for each provided batch containing their
    respective predictions, or a list of lists, one for each provided dataloader
    containing their respective predictions, where each list contains dictionaries.

    Args:
        predictions (List[Any]): Predictions returned by `Trainer.predict` method.
        dirname (str): Dirname for predictions.
        output_format (str): Output file format. It could be `json` or `csv`.
            Default to `json`.
    """

    if not predictions:
        log.warning("Predictions is empty! Saving was cancelled ...")
        return

    if output_format not in ("json", "csv"):
        raise NotImplementedError(
            f"{output_format} is not implemented! Use `json` or `csv`."
            "Or change `continuous_decoding.utils.saving.save_predictions` func logic."
        )

    path = Path(dirname) / "predictions"
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(predictions[0], dict):
        target_path = path / f"predictions.{output_format}"
        save_predictions_from_dataloader(predictions, target_path)
        log.info(f"Saved predictions to: {str(target_path)}")
        return

    elif isinstance(predictions[0], list):
        for idx, predictions_idx in enumerate(predictions):
            if not predictions_idx:
                log.warning(f"Predictions for DataLoader #{idx} is empty! Skipping...")
                continue
            target_path = path / f"predictions_{idx}.{output_format}"
            save_predictions_from_dataloader(predictions_idx, target_path)
            log.info(
                f"Saved predictions for DataLoader #{idx} to: " f"{str(target_path)}"
            )
        return

    raise Exception(
        "Passed predictions format is not supported by default!\n"
        "Make sure that it is formed correctly! It requires as List[Dict[str, Any]] type"
        "in case of predict_dataloader returns DataLoader or List[List[Dict[str, Any]]]"
        "type in case of predict_dataloader returns List[DataLoader]!\n"
        "Or change `continuous_decoding.utils.saving.save_predictions` function logic."
    )


from omegaconf import OmegaConf

def tags_to_str(cfg, location: str = "tags", join_str="__", join_kv_str="=") -> str:
    tags_ = cfg.get(location, None)
    if tags_ is None:
        return ""
    tags: Dict = OmegaConf.to_container(cfg, resolve=True)[location]
    tags_list = []
    for key, value in tags.items():
        tags_list.append(f"{key}={value}")
    return "__".join(tags_list)

