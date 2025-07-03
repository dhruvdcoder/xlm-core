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


def generator():
    for i in range(128):
        yield {"x": i}


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
        ds = datasets.Dataset.from_generator(
            generator,
            features=datasets.Features({"x": datasets.Value("int32")}),
        ).to_iterable_dataset(num_shards=4)
        self.ds = ds

    def train_dataloader(self):
        return StatefulDataLoader(
            self.ds,
            batch_size=2,
            num_workers=2,
            drop_last=True,
            persistent_workers=True,
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
