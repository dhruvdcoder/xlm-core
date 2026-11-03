"""Collators for FlexMDM (Dream-Coder) matching the reference data layout.

Reference layout (``genuine-any-order/FlexMDM``: ``data.py``, ``evals/humaneval_compare/common.py``):

    [BOS] + prompt_ids + [PAD] + answer_ids ... right-padded with PAD to ``max_length``

The PAD right after the prompt is a *kept* separator (part of the prompt mask).
For Dream's tokenizer ``pad == eos == bos == 151643``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Subset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from xlm.datamodule import Tokenizer
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def encode_flexmdm_prompt_ids(
    prompt_ids: List[int],
    *,
    max_length: int,
    pad_id: int,
    bos_id: Optional[int],
    prepend_bos: bool = True,
) -> List[int]:
    """``[bos] + prompt_ids + [pad]`` prefix; raises if it does not fit (reference behaviour)."""
    prefix: List[int] = []
    if prepend_bos:
        if bos_id is None:
            raise ValueError("prepend_bos=True but tokenizer has no bos_token_id.")
        prefix.append(int(bos_id))
    prefix.extend(int(t) for t in prompt_ids)
    prefix.append(int(pad_id))  # separator
    if len(prefix) >= max_length:
        raise ValueError(
            f"Prompt boundary {len(prefix)} >= max_length {max_length}; shorten the prompt."
        )
    return prefix


def collate_flexmdm_prompt_prefixes(
    prefixes: List[List[int]], *, max_length: int, pad_id: int
) -> Dict[str, torch.Tensor]:
    batch_size = len(prefixes)
    input_ids = torch.full((batch_size, max_length), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for i, prefix in enumerate(prefixes):
        input_ids[i, : len(prefix)] = torch.tensor(prefix, dtype=torch.long)
        attention_mask[i, : len(prefix)] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_mask": attention_mask.clone(),
    }


def encode_flexmdm_prompt_batch(
    tokenizer: Tokenizer,
    prompts: List[str],
    *,
    max_length: int,
    prepend_bos: bool = True,
) -> Dict[str, torch.Tensor]:
    """Tokenize raw prompt strings into a FlexMDM prediction batch (reference ``encode_flexmdm_prompt``)."""
    pad_id = int(tokenizer.pad_token_id)
    prefixes = [
        encode_flexmdm_prompt_ids(
            tokenizer(p, add_special_tokens=False).input_ids,
            max_length=max_length,
            pad_id=pad_id,
            bos_id=tokenizer.bos_token_id,
            prepend_bos=prepend_bos,
        )
        for p in prompts
    ]
    return collate_flexmdm_prompt_prefixes(prefixes, max_length=max_length, pad_id=pad_id)


class FlexMDMDreamPredCollator:
    """Prediction collator: ``[BOS] prompt [PAD]`` prefix, PAD-padded to ``block_size``.

    Expects examples with ``prompt_ids`` (e.g. from
    ``xlm.tasks.humaneval.humaneval_pred_preprocess_fn`` with ``strip=False, sep=""``).
    Emits ``input_ids``, ``attention_mask``, ``prompt_mask`` (and ``fixed`` as an
    alias of ``prompt_mask`` for generic logging code) plus ``pass_through_fields``.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        prepend_bos: bool = True,
        pass_through_fields: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.prepend_bos = prepend_bos
        self.pass_through_fields = (
            list(pass_through_fields)
            if pass_through_fields is not None
            else ["task_id", "sample_k"]
        )

    def __call__(self, examples: List[Mapping[str, Any]]) -> Dict[str, Any]:
        pad_id = int(self.tokenizer.pad_token_id)
        prefixes = [
            encode_flexmdm_prompt_ids(
                list(ex["prompt_ids"]),
                max_length=self.block_size,
                pad_id=pad_id,
                bos_id=self.tokenizer.bos_token_id,
                prepend_bos=self.prepend_bos,
            )
            for ex in examples
        ]
        batch: Dict[str, Any] = collate_flexmdm_prompt_prefixes(
            prefixes, max_length=self.block_size, pad_id=pad_id
        )
        batch["fixed"] = batch["prompt_mask"].long()
        for key in self.pass_through_fields:
            if key in examples[0]:
                batch[key] = [ex[key] for ex in examples]
        return batch


def print_batch_flexmdm_dream(
    batch: Mapping[str, Any],
    split: str,
    tokenizer: Tokenizer,
    dataloader_name: str = "",
) -> None:
    """Debug helper: logs the first example of a FlexMDM (Dream) batch."""
    ids = batch["input_ids"][0]
    attn = batch["attention_mask"][0].bool()
    pmask = batch.get("prompt_mask")
    n_prompt = int(pmask[0].sum()) if pmask is not None else int(attn.sum())
    text = tokenizer.decode(ids[attn], skip_special_tokens=False)
    extra = ""
    if "task_id" in batch:
        extra = f"  task_id={batch['task_id'][0]}  sample_k={batch.get('sample_k', ['?'])[0]}"
    logger.info(
        f"[FlexMDM-Dream] split={split}  dl_name={dataloader_name}  "
        f"shape={tuple(batch['input_ids'].shape)}  attended={int(attn.sum())}  "
        f"prompt_len={n_prompt}{extra}\n{text}"
    )


################################################################################
# region: pre-tokenized training shards (reference ``flexmdm.data``)

MANIFEST_FILENAME = "manifest.json"  # reference ``flexmdm.data.DEFAULT_MANIFEST_FILENAME``


class FlexMDMPretokenizedShard(Dataset):
    """Map-style view over one reference pre-tokenized shard (``*.bin`` memmaps).

    Shard metadata comes from the manifest written by ``python -m flexmdm.data tokenize``:
    ``shard_dir``, ``files``, ``dtypes``, ``shapes``, ``kept``, ``dataset_name``. Rows are
    ``[BOS] prompt [PAD] answer`` right-padded with PAD to the shard width.
    """

    ARRAY_NAMES = ("input_ids", "prompt_mask", "attention_mask", "seq_lens", "prompt_lens")

    def __init__(self, shard_metadata: Mapping[str, Any]):
        self.metadata = dict(shard_metadata)
        self.dataset_name = str(self.metadata["dataset_name"])
        self.shard_dir = os.path.expanduser(str(self.metadata["shard_dir"]))
        self.length = int(self.metadata["kept"])
        self._files: Dict[str, str] = {}
        self._dtypes: Dict[str, Any] = {}
        self._shapes: Dict[str, tuple] = {}
        for name in self.ARRAY_NAMES:
            path = str(self.metadata["files"][name])
            if not os.path.isabs(path):
                path = os.path.join(self.shard_dir, path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Shard {self.dataset_name} is missing {name}: {path}")
            self._files[name] = path
            self._dtypes[name] = np.dtype(self.metadata["dtypes"][name])
            self._shapes[name] = tuple(int(d) for d in self.metadata["shapes"][name])
        self.width = int(self._shapes["input_ids"][1])
        self._arrays: Optional[Dict[str, np.memmap]] = None  # opened lazily (fork-safe)

    def _open(self) -> Dict[str, np.memmap]:
        if self._arrays is None:
            self._arrays = {
                n: np.memmap(self._files[n], dtype=self._dtypes[n], mode="r", shape=self._shapes[n])
                for n in self.ARRAY_NAMES
            }
        return self._arrays

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = None
        return state

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        arrays = self._open()
        return {
            "input_ids": torch.as_tensor(np.array(arrays["input_ids"][index], copy=True), dtype=torch.long),
            "prompt_mask": torch.as_tensor(np.array(arrays["prompt_mask"][index], copy=True), dtype=torch.bool),
            "attention_mask": torch.as_tensor(
                np.array(arrays["attention_mask"][index], copy=True), dtype=torch.bool
            ),
            "seq_lens": torch.as_tensor(int(arrays["seq_lens"][index]), dtype=torch.long),
            "prompt_lens": torch.as_tensor(int(arrays["prompt_lens"][index]), dtype=torch.long),
            "dataset_name": self.dataset_name,
        }


def load_pretokenized_manifest(root: str) -> Dict[str, Any]:
    path = os.path.join(os.path.expanduser(root), MANIFEST_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find FlexMDM pre-tokenized manifest: {path}")
    with open(path) as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), dict):
        raise ValueError(f"Malformed manifest: {path}")
    return manifest


def build_pretokenized_concat(
    root: str,
    *,
    dataset_names: Optional[Sequence[str]] = None,
    repeat_factors: Optional[Mapping[str, int]] = None,
) -> ConcatDataset:
    """``ConcatDataset`` of shards in manifest (or ``dataset_names``) order, each repeated
    ``repeat_factors[name]`` times (reference ``_build_pretokenized_dataset``)."""
    manifest = load_pretokenized_manifest(root)
    shards = manifest["datasets"]
    names = list(dataset_names) if dataset_names else list(shards.keys())
    repeat_factors = dict(repeat_factors or {})
    datasets: List[Dataset] = []
    for name in names:
        if name not in shards:
            raise KeyError(f"Dataset {name!r} not in manifest {root}. Available: {sorted(shards)}")
        shard = FlexMDMPretokenizedShard(shards[name])
        if len(shard) == 0:
            logger.warning(f"Skipping empty shard {name}")
            continue
        repeat = max(1, int(repeat_factors.get(name, 1)))
        datasets.extend([shard] * repeat)
    if not datasets:
        raise ValueError(f"No non-empty shards under {root} for {names}")
    return ConcatDataset(datasets)


def holdout_split(dataset: Dataset, *, validation_fraction: float, seed: int):
    """Deterministic train/val split by a seeded permutation (reference ``_holdout_split``)."""
    total = len(dataset)
    val_count = int(round(total * float(validation_fraction)))
    val_count = max(1, min(val_count, total - 1))
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(int(seed))).tolist()
    return Subset(dataset, perm[val_count:]), Subset(dataset, perm[:val_count])


def collate_flexmdm_pretokenized(examples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        k: torch.stack([ex[k] for ex in examples])
        for k in ("input_ids", "attention_mask", "prompt_mask", "seq_lens", "prompt_lens")
    }
    out["dataset_name"] = [ex["dataset_name"] for ex in examples]
    return out


class FlexMDMPretokenizedDatasetManager:
    """xlm dataset manager over the reference pre-tokenized shards.

    Mirrors ``flexmdm.data.build_{train,validation}_dataloader``: concatenate shards with
    integer repeat factors, hold out ``validation_fraction`` with a seed-``validation_seed``
    permutation, then (train) shuffle with a ``DistributedSampler``-compatible sampler seeded by
    ``sampler_seed`` so the per-rank sample order matches the reference for the same world size.
    Micro-batches read consecutive slices of that order, so ``per_device_batch_size *
    accumulate_grad_batches`` samples form the same optimizer step as the reference's per-rank
    batch.

    Args:
        pretokenized_root: directory containing the reference ``manifest.json``.
        split: ``"train"`` or ``"val"`` half of the holdout split.
        dataset_names: shard names (manifest order when None).
        repeat_factors: integer up-sampling per shard (train; also val when
            ``validation_pretokenized_root`` is None, as in the reference).
        validation_fraction / validation_seed: holdout split parameters.
        sampler_seed: ``DistributedSampler`` seed (reference default 0).
        max_rows: optional cap on the split (smoke runs); applied after the split.
    """

    def __init__(
        self,
        pretokenized_root: str,
        dataloader_kwargs: Dict[str, Any],
        split: str = "train",
        dataset_names: Optional[Sequence[str]] = None,
        repeat_factors: Optional[Mapping[str, int]] = None,
        validation_fraction: float = 0.0005,
        validation_seed: int = 1,
        sampler_seed: int = 0,
        max_rows: Optional[int] = None,
        stages: Sequence[str] = ("fit",),
        full_name: Optional[str] = None,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.pretokenized_root = os.path.expanduser(pretokenized_root)
        self.dataloader_kwargs = dict(dataloader_kwargs)
        self.split = split
        self.dataset_names = list(dataset_names) if dataset_names else None
        self.repeat_factors = dict(repeat_factors or {})
        self.validation_fraction = float(validation_fraction)
        self.validation_seed = int(validation_seed)
        self.sampler_seed = int(sampler_seed)
        self.max_rows = max_rows
        self.stages = list(stages)
        self.full_name = full_name or f"flexmdm_pretokenized/{split}"
        self.is_iterable_dataset = False
        self.collator = collate_flexmdm_pretokenized
        self.dataset: Optional[Dataset] = None
        self._sampler = None

    # -- xlm DatasetManager protocol -----------------------------------------
    def prepare_data(self, manual_cache_dir: str, tokenizer: Tokenizer, num_proc: Optional[int] = None, **_):
        del manual_cache_dir, tokenizer, num_proc
        manifest = load_pretokenized_manifest(self.pretokenized_root)
        logger.info(
            f"{self.full_name}: manifest at {self.pretokenized_root} with "
            f"{manifest.get('total_written', '?')} rows across {sorted(manifest['datasets'])}"
        )

    def _build(self, block_size: Optional[int]) -> Dataset:
        full = build_pretokenized_concat(
            self.pretokenized_root, dataset_names=self.dataset_names, repeat_factors=self.repeat_factors
        )
        widths = {d.width for d in full.datasets}  # type: ignore[attr-defined]
        if block_size is not None and widths != {int(block_size)}:
            raise ValueError(f"Shard width(s) {sorted(widths)} != block_size {block_size}")
        if self.validation_fraction > 0.0:
            train, val = holdout_split(
                full, validation_fraction=self.validation_fraction, seed=self.validation_seed
            )
            ds = train if self.split == "train" else val
        else:
            if self.split == "val":
                raise ValueError("validation_fraction must be > 0 for the 'val' split")
            ds = full
        if self.max_rows is not None and len(ds) > int(self.max_rows):
            ds = Subset(ds, list(range(int(self.max_rows))))
        logger.info(f"{self.full_name}: {len(ds)} rows (full concat {len(full)})")
        return ds

    def setup(
        self,
        stage: str,
        manual_cache_dir: str,
        tokenizer: Tokenizer,
        block_size: int,
        is_ddp: bool,
        rank: int,
        world_size: int,
        num_dataset_workers: Optional[int] = None,
    ) -> None:
        del manual_cache_dir, tokenizer, is_ddp, rank, world_size, num_dataset_workers
        if stage in self.stages and self.dataset is None:
            self.dataset = self._build(block_size)

    def get_dataloader(self, type: str, is_ddp: bool, rank: int, world_size: int):
        if self.dataset is None:
            raise RuntimeError(f"{self.full_name}: setup must run before get_dataloader")
        kwargs = dict(self.dataloader_kwargs)
        kwargs.pop("shuffle", None)
        if int(kwargs.get("num_workers", 0) or 0) == 0:
            kwargs.pop("prefetch_factor", None)
            kwargs.pop("persistent_workers", None)
        if type == "train":
            # DistributedSampler semantics also for a single process (num_replicas=1), so the
            # order is a seeded randperm like the reference trainer.
            self._sampler = StatefulDistributedSampler(
                self.dataset,
                num_replicas=int(world_size) if is_ddp else 1,
                rank=int(rank) if is_ddp else 0,
                shuffle=True,
                seed=self.sampler_seed,
                drop_last=bool(kwargs.pop("drop_last", True)),
            )
            return StatefulDataLoader(self.dataset, collate_fn=self.collator, sampler=self._sampler, **kwargs)
        sampler = None
        if is_ddp and int(world_size) > 1:
            sampler = DistributedSampler(
                self.dataset, num_replicas=int(world_size), rank=int(rank), shuffle=False, drop_last=False
            )
        kwargs.pop("drop_last", None)
        return DataLoader(self.dataset, collate_fn=self.collator, sampler=sampler, drop_last=False, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        if self._sampler is not None:
            self._sampler.set_epoch(int(epoch))


# endregion
################################################################################


__all__ = [
    "encode_flexmdm_prompt_ids",
    "print_batch_flexmdm_dream",
    "collate_flexmdm_prompt_prefixes",
    "encode_flexmdm_prompt_batch",
    "FlexMDMDreamPredCollator",
    "FlexMDMPretokenizedShard",
    "FlexMDMPretokenizedDatasetManager",
    "build_pretokenized_concat",
    "holdout_split",
    "load_pretokenized_manifest",
    "collate_flexmdm_pretokenized",
    "MANIFEST_FILENAME",
]
