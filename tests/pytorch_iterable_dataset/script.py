#!/usr/bin/env python
import os
import datasets
from datasets import Dataset
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# total number of examples
DS_SIZE = 60
BATCH_SIZE = 4
NUM_WORKERS = 3
NUM_SHARDS = 8
SEED = 42

datasets.logging.set_verbosity_debug()


def dummy_generator():
    for i in range(DS_SIZE):
        yield {"data": i}


# build a HF IterableDataset
hf_ds = Dataset.from_generator(dummy_generator).to_iterable_dataset(
    num_shards=NUM_SHARDS
)

# print the shards
for i in range(hf_ds.num_shards):
    shard = hf_ds.shard(hf_ds.num_shards, i)
    examples = [ex["data"] for ex in shard]
    print(f"Shard {i}: {examples}\t length: {len(examples)}")


class LoggingDataset(IterableDataset):
    """Wrap any iterable dataset to log which worker emits which samples."""

    def __init__(self, ds):
        self.ds = ds

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        for item in self.ds:
            # print each element as it’s pulled by this worker
            print(f"[worker {worker_id}] yielding {item['data']}", flush=True)
            yield item


if __name__ == "__main__":
    # wrap and feed into a DataLoader with 2 workers
    ds = LoggingDataset(hf_ds)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    with open(f"data.txt", "w") as data_file:
        # consume everything
        for i, batch in enumerate(loader):
            data_file.write(f"Batch {i}: {batch['data'].tolist()}\n")

# python -u script.py
