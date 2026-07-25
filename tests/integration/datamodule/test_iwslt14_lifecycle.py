"""Lifecycle smoke for IWSLT14 LocalDatasetManager + preprocess + cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import datasets
import pytest

from xlm.datamodule import LocalDatasetManager
from xlm.tasks.iwslt14_de_en import iwslt14_mt_preprocess_fn


pytestmark = [pytest.mark.integration]


class _MockTok:
    pad_token_id = 0
    unk_token_id = 1
    bos_token_id = 2
    eos_token_id = 3
    mask_token_id = 4

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        del add_special_tokens
        return [10 + (ord(c) % 30) for c in text] if text else []


class _PromptCollator:
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "prompt_ids": [e["prompt_ids"] for e in examples],
            "input_ids": [e["input_ids"] for e in examples],
            "n": len(examples),
        }


def _write_fixture_parquet(path: Path, n: int = 8) -> None:
    rows = {
        "id": [f"id-{i}" for i in range(n)],
        "source_text": [f"quelle {i}" for i in range(n)],
        "target_text": [f"target {i}" for i in range(n)],
        "source_raw": [f"Q{i}" for i in range(n)],
        "target_raw": [f"T{i}" for i in range(n)],
        "origin": ["train"] * n,
        "origin_index": list(range(n)),
        "segment_id": [None] * n,
        "source_word_count": [2] * n,
        "target_word_count": [2] * n,
        "retained_index": list(range(1, n + 1)),
        "split": ["train"] * n,
        "component": [None] * n,
        "source_language": ["de"] * n,
        "target_language": ["en"] * n,
    }
    datasets.Dataset.from_dict(rows).to_parquet(str(path))


def _make_manager(parquet_path: Path) -> LocalDatasetManager:
    return LocalDatasetManager(
        collator=_PromptCollator(),
        full_name="local/iwslt14_de_en_rdm_text_v1/train",
        full_name_debug="local/iwslt14_de_en_rdm_text_v1/train",
        dataloader_kwargs={
            "batch_size": 2,
            "num_workers": 0,
            "shuffle": False,
            "pin_memory": False,
            "drop_last": False,
        },
        ds_type="parquet",
        load_kwargs={"data_files": str(parquet_path)},
        preprocess_function="xlm.tasks.iwslt14_de_en.iwslt14_mt_preprocess_fn",
        on_the_fly_processor="xlm.datamodule.token_ids_to_input_ids_and_prompt_ids",
        on_the_fly_group_processor=None,
        columns_to_remove=None,
        columns_to_keep=["id", "source_text", "target_text", "target_raw"],
        filter_suffix="tok_975aedcb168b_rawref",
        stages=["fit"],
        iterable_dataset_shards=None,
        shuffle_buffer_size=None,
        use_manual_cache=True,
    )


def test_iwslt14_prepare_setup_batch(tmp_path: Path):
    parquet_path = tmp_path / "train.parquet"
    _write_fixture_parquet(parquet_path)
    cache_dir = tmp_path / "cache"
    tok = _MockTok()
    mgr = _make_manager(parquet_path)

    mgr.prepare_data(manual_cache_dir=str(cache_dir), tokenizer=tok, num_proc=1)
    assert mgr._check_cache(str(cache_dir))

    mgr.setup(
        stage="fit",
        manual_cache_dir=str(cache_dir),
        tokenizer=tok,
        block_size=64,
        is_ddp=False,
        rank=0,
        world_size=1,
    )
    dl = mgr.get_dataloader(type="train", is_ddp=False, rank=0, world_size=1)
    batch = next(iter(dl))
    assert batch["n"] == 2
    assert len(batch["prompt_ids"][0]) > 0
    assert len(batch["input_ids"][0]) > 0


def test_iwslt14_preprocess_determinism_num_proc(tmp_path: Path):
    parquet_path = tmp_path / "train.parquet"
    _write_fixture_parquet(parquet_path, n=6)
    tok = _MockTok()

    def _encode_all(num_proc: int) -> List[List[int]]:
        cache = tmp_path / f"cache_np{num_proc}"
        mgr = _make_manager(parquet_path)
        mgr.prepare_data(
            manual_cache_dir=str(cache), tokenizer=tok, num_proc=num_proc
        )
        cached = datasets.load_from_disk(str(mgr._get_cache_dir(str(cache))))
        return list(cached["prompt_token_ids"])

    assert _encode_all(1) == _encode_all(2)


def test_iwslt14_preprocess_fn_matches_direct_encode():
    tok = _MockTok()
    row = {"source_text": "alpha", "target_text": "beta"}
    out = iwslt14_mt_preprocess_fn(dict(row), tok)
    assert out["prompt_token_ids"] == tok.encode("alpha")
    assert out["input_token_ids"] == tok.encode("beta")
