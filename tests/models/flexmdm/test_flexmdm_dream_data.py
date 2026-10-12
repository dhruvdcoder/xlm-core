"""FlexMDMPretokenizedDatasetManager: reads reference ``.bin`` shards and reproduces the
reference train/val split and per-rank sample order."""

import json
import os

import numpy as np
import pytest
import torch

from flexmdm.datamodule_flexmdm_dream import (
    MANIFEST_FILENAME,
    FlexMDMPretokenizedDatasetManager,
    FlexMDMPretokenizedShard,
    build_pretokenized_concat,
    holdout_split,
)

from .ref_utils import import_reference_submodule

WIDTH = 16
PAD, BOS = 3, 1


def _write_shard(root, name, n, seed):
    """Write a shard in the reference writer's layout (int32/uint8/uint16 raw files)."""
    rng = np.random.default_rng(seed)
    shard_dir = os.path.join(root, name)
    os.makedirs(shard_dir)
    ids = np.full((n, WIDTH), PAD, dtype=np.int32)
    pm = np.zeros((n, WIDTH), dtype=np.uint8)
    am = np.zeros((n, WIDTH), dtype=np.uint8)
    seq_lens = np.zeros(n, dtype=np.uint16)
    prompt_lens = np.zeros(n, dtype=np.uint16)
    for i in range(n):
        p = int(rng.integers(1, 5))
        a = int(rng.integers(1, WIDTH - p - 2))
        row = [BOS] + rng.integers(10, 100, size=p).tolist() + [PAD] + rng.integers(10, 100, size=a).tolist()
        ids[i, : len(row)] = row
        am[i, : len(row)] = 1
        pm[i, : p + 2] = 1
        seq_lens[i], prompt_lens[i] = len(row), p + 2
    files = {}
    for k, arr in {"input_ids": ids, "prompt_mask": pm, "attention_mask": am, "seq_lens": seq_lens, "prompt_lens": prompt_lens}.items():
        files[k] = os.path.join(shard_dir, f"{k}.bin")
        arr.tofile(files[k])
    return {
        "dataset_name": name,
        "shard_dir": shard_dir,
        "kept": n,
        "files": files,
        "dtypes": {"input_ids": "int32", "prompt_mask": "uint8", "attention_mask": "uint8", "seq_lens": "uint16", "prompt_lens": "uint16"},
        "shapes": {"input_ids": [n, WIDTH], "prompt_mask": [n, WIDTH], "attention_mask": [n, WIDTH], "seq_lens": [n], "prompt_lens": [n]},
    }


@pytest.fixture
def root(tmp_path):
    root = str(tmp_path / "pretok")
    os.makedirs(root)
    datasets = {name: _write_shard(root, name, n, i) for i, (name, n) in enumerate([("a", 37), ("b", 11), ("c", 23)])}
    with open(os.path.join(root, MANIFEST_FILENAME), "w") as f:
        json.dump({"datasets": datasets, "total_written": 71}, f)
    return root


def test_shard_reader_matches_reference(root):
    ref_data = import_reference_submodule("data")
    with open(os.path.join(root, MANIFEST_FILENAME)) as f:
        meta = json.load(f)["datasets"]["a"]
    ours, ref = FlexMDMPretokenizedShard(meta), ref_data.FlexMDMTokenizedShardDataset(meta)
    assert len(ours) == len(ref) == 37
    for i in (0, 5, 36, -1):
        a, b = ours[i], ref[i]
        for k in ("input_ids", "prompt_mask", "attention_mask", "seq_lens", "prompt_lens"):
            assert a[k].dtype == b[k].dtype, k
            assert torch.equal(a[k], b[k]), k
        assert a["dataset_name"] == b["dataset_name"] == "a"
    row = ours[0]
    assert row["input_ids"][0] == BOS and row["prompt_mask"][: row["prompt_lens"]].all()


def test_concat_split_and_order_match_reference(root):
    ref_data = import_reference_submodule("data")
    repeat = {"a": 2, "b": 1, "c": 2}
    data_cfg = {"pretokenized_root": root, "dataloader_repeat_factors": repeat, "validation_fraction": 0.1,
                "validation_seed": None, "num_workers": 0, "pin_memory": False, "drop_last": True,
                "train_batch_size": 8, "train_batch_size_is_global": True, "micro_batch_size_per_gpu": 2, "validation_batch_size": 4}
    cfg = {"data": data_cfg, "trainer": {"seed": 1}}

    full = build_pretokenized_concat(root, repeat_factors=repeat)
    assert len(full) == 37 * 2 + 11 + 23 * 2
    train, val = holdout_split(full, validation_fraction=0.1, seed=1)

    for rank, world in ((0, 2), (1, 2)):
        ref_dl = ref_data.build_train_dataloader(config=cfg, rank=rank, world_size=world)
        assert len(ref_dl.dataset) == len(train)
        ref_dl.sampler.set_epoch(0)
        ref_rows = [r for b in ref_dl for r in b["input_ids"]]  # per-rank batch 4

        mgr = FlexMDMPretokenizedDatasetManager(
            root, {"batch_size": 2, "num_workers": 0, "drop_last": True}, split="train",
            repeat_factors=repeat, validation_fraction=0.1, validation_seed=1, sampler_seed=0,
        )
        mgr.setup("fit", "", None, WIDTH, True, rank, world)
        dl = mgr.get_dataloader("train", True, rank, world)
        mgr.set_epoch(0)
        our_rows = [r for b in dl for r in b["input_ids"]]  # micro-batch 2, same stream
        assert len(our_rows) == len(ref_rows) > 0
        assert all(torch.equal(x, y) for x, y in zip(our_rows, ref_rows))

    # validation half: same membership as the reference holdout
    ref_val = ref_data.build_validation_dataloader(config=cfg, rank=0, world_size=1)
    vmgr = FlexMDMPretokenizedDatasetManager(
        root, {"batch_size": 4, "num_workers": 0}, split="val", repeat_factors=repeat,
        validation_fraction=0.1, validation_seed=1,
    )
    vmgr.setup("fit", "", None, WIDTH, False, 0, 1)
    vdl = vmgr.get_dataloader("val", False, 0, 1)
    ref_val_rows = [r for b in ref_val for r in b["input_ids"]]
    our_val_rows = [r for b in vdl for r in b["input_ids"]]
    # reference val loader keeps drop_last=True (batch 4 here), ours keeps every row
    assert len(our_val_rows) == len(val) and len(ref_val_rows) == len(val) // 4 * 4
    assert all(torch.equal(x, y) for x, y in zip(our_val_rows, ref_val_rows))


def test_block_size_mismatch_raises(root):
    mgr = FlexMDMPretokenizedDatasetManager(root, {"batch_size": 2}, validation_fraction=0.0)
    with pytest.raises(ValueError, match="block_size"):
        mgr.setup("fit", "", None, WIDTH + 1, False, 0, 1)
    mgr.setup("fit", "", None, WIDTH, False, 0, 1)
    batch = next(iter(mgr.get_dataloader("train", False, 0, 1)))
    assert batch["input_ids"].shape == (2, WIDTH)
    assert batch["prompt_mask"].dtype == torch.bool and (batch["prompt_mask"] <= batch["attention_mask"]).all()
