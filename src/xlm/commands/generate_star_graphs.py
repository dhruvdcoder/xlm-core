# %%
import os
from pathlib import Path
from typing import Optional
from xlm.tasks.star import (
    SimpleStarGraphGenerator,
    AsymmetricVariableArmlengthStarGraphGenerator,
)
import datasets
from xlm.utils.seed import seed_everything_simple
import dotenv

dotenv.load_dotenv(override=True)
dotenv.load_dotenv(".secrets.env", override=True)
seed_everything_simple(42)

datasets.logging.set_verbosity_info()

# start_small i.e., the "easy" setting
SIZES = {
    "train": 50000,
    "validation": 100,
    "test": 5000,
}

STAR_EASY = {
    "constructor": SimpleStarGraphGenerator,
    "kwargs": {
        "degree": 3,
        "pathlength": 5,
        "vocab_size": 20,
    },
    "name": "star-small",
}

STAR_MEDIUM = {
    "constructor": AsymmetricVariableArmlengthStarGraphGenerator,
    "kwargs": {
        "degree": 3,
        "min_pathlength": 3,
        "max_pathlength": 6,
        "min_plan_length": 2,
        "vocab_size": 20,
    },
    "name": "star-medium",
}

STAR_HARD = {
    "constructor": AsymmetricVariableArmlengthStarGraphGenerator,
    "kwargs": {
        "degree": 5,
        "min_pathlength": 6,
        "max_pathlength": 12,
        "min_plan_length": 5,
        "vocab_size": 56,
    },
    "name": "star-hard",
}


# %%
def dump_dataset(
    settings: dict, path: Optional[Path] = None, push_to_hub: bool = False
):
    name = settings["name"]
    datasets_: dict[str, datasets.Dataset] = {}
    for split, size in SIZES.items():
        generator = settings["constructor"](
            count=size,
            **settings["kwargs"],
        )

        dataset = datasets.Dataset.from_generator(
            generator.graphs,
            keep_in_memory=True,
            num_proc=1,
        )
        datasets_[split] = dataset

    dataset_dict = datasets.DatasetDict(datasets_)
    if push_to_hub:
        dataset_dict.push_to_hub(
            f"dhruveshpatel/{name}", token=os.getenv("HF_HUB_KEY")
        )
    if path is not None:
        dataset_dict.save_to_disk(path)
    return dataset


# %%
if __name__ == "__main__":
    # %%
    dump_dataset(STAR_EASY, path=None, push_to_hub=True)
    dump_dataset(STAR_MEDIUM, path=None, push_to_hub=True)
    dump_dataset(STAR_HARD, path=None, push_to_hub=True)

# %%
