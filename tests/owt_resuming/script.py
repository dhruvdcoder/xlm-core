#!/usr/bin/env python
import argparse
import os
import time
from typing import Dict, List
import torch
import lightning as pl
from torch.utils.data import DataLoader
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
import datasets
from transformers import AutoTokenizer
from more_itertools import flatten, chunked
from torchdata.stateful_dataloader import StatefulDataLoader
from lightning.pytorch.callbacks.on_exception_checkpoint import (
    OnExceptionCheckpoint,
)

datasets.logging.set_verbosity_debug()

import logging


# class SuppressBufferedShuffleWarning(logging.Filter):
#    def filter(self, record):
#        # Check if the log message contains the specific text you want to ignore
#        if "shuffle buffer will be refilled" in record.getMessage():
#            return False  # Do not log this message
#        return True
#
#
## in the base datamodule.py
# _logger = logging.getLogger("datasets.iterable_dataset")
# _logger.addFilter(SuppressBufferedShuffleWarning())


def dummy_generator():
    # Generate 60 examples: integers from $0$ to $59$
    # 64 sequences of different lengths
    dataset = [
        list(range(3, 10)),
        list(range(10, 15)),
        list(range(15, 21)),
        list(range(21, 27)),
        list(range(27, 31)),
        list(range(31, 36)),
        list(range(36, 45)),
        list(range(45, 50)),
    ]
    for i in range(8):
        for j, ids in enumerate(dataset):
            yield {"token_ids": [idx + i * 50 for idx in ids]}


def group_texts(
    examples: Dict[str, List[List[int]]],
    block_size: int,
    eos_token_id: int,
    bos_token_id: int,
    pad_token_id: int,
) -> Dict[str, List[List[int]]]:
    real_block_size = block_size - 2  # make space for bos and eos
    # colapse the sequences into a single list of tokens and then create blocks of real_block_size
    input_ids = []
    attention_mask = []
    for block in chunked(flatten(examples["token_ids"]), real_block_size):
        s = [bos_token_id] + list(block) + [eos_token_id]
        ls = len(s)
        attn = [True] * ls
        s += [pad_token_id] * (block_size - ls)
        attn += [False] * (block_size - ls)
        input_ids.append(s)
        attention_mask.append(attn)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def collate_fn(batch):
    return {
        "input_ids": torch.tensor(
            [item["input_ids"] for item in batch], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [item["attention_mask"] for item in batch], dtype=torch.long
        ),
    }


class DummyModule(pl.LightningModule):

    def __init__(self, interrupt_step: int):
        super().__init__()
        # A dummy linear layer (not used for actual computation)
        self.layer = torch.nn.Linear(1, 1)
        self.ds = None
        self.prepare_data_per_node = False
        self.interrupt_step = interrupt_step

    def on_train_start(self):
        # This hook is called once training begins on each process.
        print(f"[Rank {self.global_rank}] Training started.", flush=True)
        self.data_file = open(f"data_{self.global_rank}.txt", "w")

    def on_train_end(self):
        self.data_file.close()

    def training_step(self, batch, batch_idx):
        # Print batch information to verify data loading.
        print(
            f"[Rank {self.global_rank}] Training step {self.trainer.global_step}, epoch {self.trainer.current_epoch}",
            flush=True,
        )
        # time.sleep(1)
        interrupt_step = (
            self.interrupt_step if self.interrupt_step is not None else -1
        )
        if self.trainer.global_step == interrupt_step:
            raise Exception("Interrupt")
        self.data_file.write(f"batch: {batch['input_ids']}\n")
        # print("batch", batch, flush=True)
        loss = torch.tensor(0.0, requires_grad=True)
        return loss

    def on_train_epoch_start(self):
        epoch = self.trainer.current_epoch
        print(
            f"[Rank {self.global_rank}] Training epoch {epoch} started at step {self.trainer.global_step}",
            flush=True,
        )
        self.data_file.write(
            f"[Rank {self.global_rank}] Training epoch {epoch} started at step {self.trainer.global_step}\n"
        )

    def configure_optimizers(self):
        # Return a dummy optimizer.
        return torch.optim.SGD(self.parameters(), lr=0.001)


class DM(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.ds = None
        self.prepare_data_per_node = False

    def set_epoch(self, epoch: int):
        self.ds.set_epoch(epoch)

    def prepare_data(self):
        # download the dataset
        if os.path.exists("dataset"):
            print("Dataset already exists")
            return
        dataset = Dataset.from_generator(dummy_generator)
        dataset = dataset.map(
            group_texts,
            batched=True,
            batch_size=5,
            fn_kwargs={
                "block_size": 5,
                "eos_token_id": 1,
                "bos_token_id": 0,
                "pad_token_id": 2,
            },
            remove_columns=["token_ids"],
        )
        # save the dataset
        dataset.save_to_disk("dataset", num_shards=4)

    def setup(self, stage: str):
        # load the dataset
        print("Setup")
        ds = (
            datasets.load_from_disk("dataset").to_iterable_dataset(
                num_shards=4
            )
            # .shuffle(seed=42)
        )
        ds = split_dataset_by_node(
            ds,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
        )
        self.ds = ds

    def train_dataloader(self):
        print(
            f"[Rank {self.trainer.global_rank}] Preparing train_dataloader...",
            flush=True,
        )
        rank = self.trainer.global_rank
        print(
            f"[Rank {rank}] Global rank: {self.trainer.global_rank}",
            flush=True,
        )
        world_size = self.trainer.world_size
        print(f"[Rank {rank}] World size: {world_size}", flush=True)
        return StatefulDataLoader(
            self.ds,
            batch_size=2,
            num_workers=2,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=False,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--interrupt-step", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Starting Lightning training", flush=True)
    # Optionally, print some SLURM environment info for debugging.
    print(f"SLURM_NNODES: {os.environ.get('SLURM_NNODES', '1')}", flush=True)

    # Determine the number of nodes from SLURM (defaulting to 1 if not set)

    interrupt_step = args.interrupt_step
    num_nodes = int(os.environ.get("SLURM_NNODES", "1"))
    model = DummyModule(interrupt_step=interrupt_step)

    dm = DM()
    on_exception = OnExceptionCheckpoint(
        dirpath="checkpoints",
        filename="on_exception",
    )

    # Configure the Trainer to use distributed data parallel (DDP).
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy=(
            "ddp" if num_nodes > 1 else "auto"
        ),  # Use DDP strategy for multi-node training.
        num_nodes=num_nodes,
        max_epochs=5,
        logger=False,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        callbacks=[on_exception],
    )
    resume = args.resume
    # resume (uncomment to resume)
    if resume:
        trainer.fit(
            model, datamodule=dm, ckpt_path="checkpoints/on_exception.ckpt"
        )
    # train
    else:
        trainer.fit(
            model,
            datamodule=dm,
        )
