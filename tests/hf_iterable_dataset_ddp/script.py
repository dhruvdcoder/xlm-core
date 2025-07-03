#!/usr/bin/env python
import os
import torch
import lightning as pl
from torch.utils.data import DataLoader
import datasets
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from lightning_utilities.core.rank_zero import (
    rank_zero_info,
    rank_prefixed_message,
)
from collections import Counter

## region: all works out
# NUM_WORKERS = 2
# BATCH_SIZE = 2
# NUM_SHARDS = 8
# DS_SIZE = 60
# SEED = 42
# SHUFFLE = False
# DROP_LAST = True
## endregion: all works out

# region: shards_per_node is not divisible by num_workers
NUM_WORKERS = 2
BATCH_SIZE = 2
NUM_SHARDS = 8
DS_SIZE = 60
SEED = 42
SHUFFLE = False
DROP_LAST = True
# endregion: all works out

datasets.logging.set_verbosity_debug()

# seed everything
pl.seed_everything(SEED)


def count_total_examples(ds):
    return sum(1 for _ in ds)


def print_shards_info(ds, rank=None):
    shard_info = {}
    for i in range(ds.num_shards):
        shard = ds.shard(ds.num_shards, i)
        count = sum(1 for _ in shard)
        shard_info[i] = count
    # Build histogram: count -> number of shards with that count
    hist = Counter(shard_info.values())
    lines = []
    for num_examples, num_shards in sorted(hist.items()):
        lines.append(f"{num_shards} shard(s) with {num_examples} examples")
    msg = ", ".join(lines)
    if len(lines) == 0 and rank is not None:
        rank_prefixed_message("No shards found", rank=rank)
    if rank is None:
        rank_zero_info(f"Shard histogram: {msg}")
    else:
        rank_prefixed_message(f"Shard histogram: {msg}", rank=rank)


def dummy_generator():
    # Generate 60 examples: integers from $0$ to $59$
    for i in range(DS_SIZE):
        yield {"data": i}


class DummyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # A dummy linear layer (not used for actual computation)
        self.layer = torch.nn.Linear(1, 1)
        self.ds = None
        self.data_file = None
        self.examples_seen_in_epoch = 0

    def on_train_start(self):
        # This hook is called once training begins on each process.
        print(f"[Rank {self.global_rank}] Training started.", flush=True)
        if self.data_file is None:
            self.data_file = open(f"data_{self.global_rank}.txt", "w")

    def on_train_end(self):
        if self.data_file is not None:
            self.data_file.close()

    def training_step(self, batch, batch_idx):
        # store the number of examples seen
        self.examples_seen_in_epoch += len(batch["data"])
        # Print batch information to verify data loading.
        print(
            f"\n[Rank {self.global_rank}] Training step, epoch {self.trainer.current_epoch}, batch {batch_idx}: {batch['data']}",
            flush=True,
        )
        self.data_file.write(
            f"[Rank {self.global_rank}] Training step, epoch {self.trainer.current_epoch}, batch {batch_idx}: {batch['data']}\n"
        )
        # Compute a dummy loss (here, simply a constant tensor)
        loss = torch.tensor(0.0, requires_grad=True)
        return loss

    def on_train_epoch_end(self):
        epoch = self.trainer.current_epoch
        # print previous epoch's examples seen
        print(
            rank_prefixed_message(
                f"Examples seen in epoch {epoch}: {self.examples_seen_in_epoch}",
                rank=self.global_rank,
            )
        )

    def on_train_epoch_start(self):
        epoch = self.trainer.current_epoch
        print(
            f"[Rank {self.global_rank}] Training epoch {epoch} started.",
            flush=True,
        )
        self.data_file.write(
            f"[Rank {self.global_rank}] Training epoch {epoch} started.\n"
        )
        self.ds.set_epoch(epoch)
        self.examples_seen_in_epoch = 0

    def train_dataloader(self):
        # Get rank and world size. Lightning sets these environment variables for DDP.
        # The trainer is set quite early in the start-up so global_rank becomes available
        rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        print(f"[Rank {rank}] Global rank: {self.global_rank}", flush=True)
        print(
            f"[Rank {self.global_rank}] Preparing train_dataloader...",
            flush=True,
        )
        # the dataloader might be requested before the on_train_start hook is called
        if self.data_file is None:
            print(f"[Rank {rank}] Creating data file", flush=True)
            self.data_file = open(f"data_{self.global_rank}.txt", "w")
        # Create the dummy dataset and convert it to an iterable dataset.
        ds = Dataset.from_generator(dummy_generator).to_iterable_dataset(
            num_shards=NUM_SHARDS
        )
        # Print global shards info
        print_shards_info(ds)

        print(f"[Rank {rank}] World size: {world_size}", flush=True)
        # Split the dataset across nodes.
        if SHUFFLE:
            ds = ds.shuffle(
                buffer_size=10, seed=SEED
            )  # you set the seed, it should be the same across ranks
        ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
        # Print local shards info
        print_shards_info(ds, rank=rank)
        total_examples = count_total_examples(ds)
        print(
            rank_prefixed_message(
                f"Total examples: {total_examples}",
                rank=rank,
            )
        )
        self.ds = ds
        print(f"[Rank {rank}] Dataset split: {ds}", flush=True)
        return DataLoader(
            self.ds, batch_size=2, num_workers=2, drop_last=DROP_LAST
        )

    def configure_optimizers(self):
        # Return a dummy optimizer.
        return torch.optim.SGD(self.parameters(), lr=0.001)


if __name__ == "__main__":
    print("Starting Lightning training", flush=True)
    # Optionally, print some SLURM environment info for debugging.
    print(f"SLURM_NNODES: {os.environ.get('SLURM_NNODES', '1')}", flush=True)

    # Determine the number of nodes from SLURM (defaulting to 1 if not set)
    num_nodes = int(os.environ.get("SLURM_NNODES", "1"))

    # Configure the Trainer to use distributed data parallel (DDP).
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy="ddp",  # Use DDP strategy for multi-node training.
        num_nodes=num_nodes,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )
    model = DummyModule()
    trainer.fit(model)
