#!/usr/bin/env python
"""Merge the released FlexMDM (Dream-Coder-7B) checkpoint into a single xlm-core checkpoint.

The authors' release (``yuyuanchen0/flexmdm``) ships a stock Dream HF checkpoint
(sharded ``model-*.safetensors`` with ``model.*`` / ``lm_head.weight`` keys) plus a
separate ``flexmdm_extras.pt`` holding the FlexMDM additions
(``insertion_head.*``, ``time_mlp.*``, ``temb_mods.{i}.*``). xlm-core's
``hub.repo_id`` loader expects *one* state dict in the model's own key layout, so
this script

1. streams the backbone shards (no full-model copy in RAM),
2. remaps the extras into ``flexmdm.model_flexmdm_dream.FlexMDMDreamModel`` keys
   (``temb_mods.{i}.*`` -> ``model.layers.{i}.rms_norm_modulations.modulation.*``),
3. verifies the merged key set against the model class (built on the meta device),
4. writes sharded safetensors + ``model.safetensors.index.json``, copies the
   tokenizer / config files, writes a provenance model card, and optionally pushes
   the folder to the Hub.

CPU only; peak RAM is roughly one output shard (``--max-shard-size``).

Example::

    python xlm-models/flexmdm/scripts/merge_flexmdm_dream_checkpoint.py \
        --output-dir ~/checkpoints/flexmdm-dream-coder-7b \
        --push-to dhruveshpatel/dream_flexmdm

The canonical merged copy lives at ``dhruveshpatel/dream_flexmdm`` (built from
``yuyuanchen0/flexmdm`` @ 1080b5f4fd8f4b3f7699736c67b7bfbfdc2e5d30).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:  # make ``flexmdm`` (xlm-models) importable
    sys.path.insert(0, str(ROOT.parent))

from flexmdm.configuration_flexmdm_dream import FlexMDMDreamConfig  # noqa: E402
from flexmdm.model_flexmdm_dream import FlexMDMDreamModel  # noqa: E402

SOURCE_REPO = "yuyuanchen0/flexmdm"
EXTRAS_FILE = "flexmdm_extras.pt"
INDEX_FILE = "model.safetensors.index.json"
SKIP_COPY = {".gitattributes", "README.md", EXTRAS_FILE, INDEX_FILE}
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_size(s: str) -> int:
    s = s.strip().upper()
    for suffix, mult in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


def download_source(repo_id: str, revision: str | None, cache_dir: str | None) -> Tuple[Path, str]:
    from huggingface_hub import snapshot_download, HfApi

    sha = HfApi().model_info(repo_id, revision=revision).sha
    local = snapshot_download(repo_id=repo_id, revision=sha, cache_dir=cache_dir)
    return Path(local), sha


def expected_state_dict_keys(config: FlexMDMDreamConfig) -> Dict[str, Tuple[int, ...]]:
    with torch.device("meta"):
        model = FlexMDMDreamModel(config, max_length=768)
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def plan_shards(entries: List[Tuple[str, int]], max_bytes: int) -> List[List[str]]:
    shards: List[List[str]] = [[]]
    used = 0
    for key, nbytes in entries:
        if shards[-1] and used + nbytes > max_bytes:
            shards.append([])
            used = 0
        shards[-1].append(key)
        used += nbytes
    return shards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE_REPO)
    ap.add_argument("--revision", default=None, help="source revision (default: main, pinned to its sha)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    ap.add_argument("--max-shard-size", default="5GB")
    ap.add_argument("--push-to", default=None, help="Hub repo id to upload the merged folder to")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f"{out} is not empty (use --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    dtype = DTYPES[args.dtype]

    print(f"[1/5] downloading {args.source} ...", flush=True)
    src, sha = download_source(args.source, args.revision, args.cache_dir)
    print(f"      snapshot {src} @ {sha}")

    # ---- expected keys from the model class (meta device, no memory) ----
    hf_cfg = json.loads((src / "config.json").read_text())
    config = FlexMDMDreamConfig(
        vocab_size=hf_cfg["vocab_size"],
        hidden_size=hf_cfg["hidden_size"],
        intermediate_size=hf_cfg["intermediate_size"],
        num_hidden_layers=hf_cfg["num_hidden_layers"],
        num_attention_heads=hf_cfg["num_attention_heads"],
        num_key_value_heads=hf_cfg["num_key_value_heads"],
        max_position_embeddings=hf_cfg["max_position_embeddings"],
        rms_norm_eps=hf_cfg["rms_norm_eps"],
        rope_theta=hf_cfg["rope_theta"],
        tie_word_embeddings=hf_cfg["tie_word_embeddings"],
        mask_token_id=hf_cfg["mask_token_id"],
        pad_token_id=hf_cfg["pad_token_id"],
        bos_token_id=hf_cfg["bos_token_id"],
        eos_token_id=hf_cfg["eos_token_id"],
        use_cache=False,
    )
    expected = expected_state_dict_keys(config)

    # ---- source inventory: backbone shards (lazy) + extras (mmap) ----
    print("[2/5] indexing source tensors ...", flush=True)
    index = json.loads((src / INDEX_FILE).read_text())
    backbone_loc: Dict[str, str] = index["weight_map"]  # key -> shard file
    entries: List[Tuple[str, int]] = []
    handles: Dict[str, "safe_open"] = {}
    for shard in sorted(set(backbone_loc.values())):
        handles[shard] = safe_open(str(src / shard), framework="pt", device="cpu")
    for key, shard in backbone_loc.items():
        sl = handles[shard].get_slice(key)
        shape = tuple(sl.get_shape())
        if key not in expected:
            sys.exit(f"backbone key {key!r} not in {FlexMDMDreamModel.__name__} state dict")
        if expected[key] != shape:
            sys.exit(f"shape mismatch for {key}: source {shape} vs model {expected[key]}")
        numel = 1
        for d in shape:
            numel *= d
        entries.append((key, numel * dtype.itemsize))

    extras_raw = torch.load(src / EXTRAS_FILE, map_location="cpu", mmap=True, weights_only=True)
    extras = FlexMDMDreamModel.remap_extras_state_dict(extras_raw)
    for key, tensor in extras.items():
        if key not in expected:
            sys.exit(f"extras key {key!r} not in {FlexMDMDreamModel.__name__} state dict")
        if expected[key] != tuple(tensor.shape):
            sys.exit(f"shape mismatch for {key}: extras {tuple(tensor.shape)} vs model {expected[key]}")
        entries.append((key, tensor.numel() * dtype.itemsize))

    merged_keys = {k for k, _ in entries}
    missing = sorted(set(expected) - merged_keys)
    if missing:
        sys.exit(f"merged checkpoint would miss {len(missing)} keys, e.g. {missing[:5]}")
    print(f"      backbone tensors: {len(backbone_loc)}  extras tensors: {len(extras)}  "
          f"total: {len(entries)} (model expects {len(expected)})")

    # ---- write output shards ----
    print("[3/5] writing shards ...", flush=True)
    shards = plan_shards(entries, parse_size(args.max_shard_size))
    n = len(shards)
    weight_map: Dict[str, str] = {}
    total_size = 0
    for i, keys in enumerate(shards, start=1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors" if n > 1 else "model.safetensors"
        tensors: Dict[str, torch.Tensor] = {}
        for key in keys:
            if key in extras:
                t = extras[key]
            else:
                t = handles[backbone_loc[key]].get_tensor(key)
            t = t.to(dtype).contiguous()
            tensors[key] = t
            weight_map[key] = name
            total_size += t.numel() * t.element_size()
        save_file(tensors, str(out / name), metadata={"format": "pt"})
        print(f"      wrote {name} ({sum(t.numel() * t.element_size() for t in tensors.values())/1e9:.2f} GB, {len(keys)} tensors)")
        del tensors
    if n > 1:
        (out / INDEX_FILE).write_text(
            json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
        )

    # ---- verify by re-reading headers ----
    print("[4/5] verifying ...", flush=True)
    seen: Dict[str, Tuple[int, ...]] = {}
    for name in sorted(set(weight_map.values())):
        with safe_open(str(out / name), framework="pt", device="cpu") as f:
            for key in f.keys():
                seen[key] = tuple(f.get_slice(key).get_shape())
    if seen != expected:
        extra = sorted(set(seen) - set(expected))
        miss = sorted(set(expected) - set(seen))
        bad = [k for k in seen if k in expected and seen[k] != expected[k]]
        sys.exit(f"verification failed: extra={extra[:5]} missing={miss[:5]} shape_mismatch={bad[:5]}")
    # spot-check a few values against the sources
    with safe_open(str(out / weight_map["lm_head.weight"]), framework="pt", device="cpu") as f:
        a = f.get_tensor("lm_head.weight")
    b = handles[backbone_loc["lm_head.weight"]].get_tensor("lm_head.weight").to(dtype)
    assert torch.equal(a, b), "lm_head mismatch"
    k0 = "model.layers.0.rms_norm_modulations.modulation.weight"
    with safe_open(str(out / weight_map[k0]), framework="pt", device="cpu") as f:
        assert torch.equal(f.get_tensor(k0), extras[k0].to(dtype)), "extras mismatch"
    print(f"      OK: {len(seen)} tensors, {total_size/1e9:.2f} GB")

    # ---- copy tokenizer/config files + model card ----
    for p in src.iterdir():
        if p.is_file() and p.name not in SKIP_COPY and not p.name.endswith(".safetensors"):
            shutil.copy2(p, out / p.name)
    cfg_out = dict(hf_cfg)
    cfg_out.update(
        {
            "architectures": ["FlexMDMDreamModel"],
            "auto_map": None,
            "use_time_embedding": True,
            "time_conditioning": "flexmdm_shared2",
            "time_scale": 1000.0,
            "time_max_period": 10000,
            "torch_dtype": args.dtype,
            "_source_repo": args.source,
            "_source_revision": sha,
        }
    )
    (out / "config.json").write_text(json.dumps(cfg_out, indent=2) + "\n")
    (out / "README.md").write_text(
        f"""---
license: apache-2.0
base_model: {args.source}
tags: [flexmdm, dream-coder, diffusion-lm, xlm-core]
---

# FlexMDM (Dream-Coder-7B) — merged checkpoint for xlm-core

Merged copy of [{args.source}](https://huggingface.co/{args.source}) (revision `{sha}`):
the Dream-Coder backbone shards and `flexmdm_extras.pt` (insertion head, time MLP,
per-layer AdaLN modulations) combined into one sharded safetensors state dict in the
key layout of `flexmdm.model_flexmdm_dream.FlexMDMDreamModel` (xlm-core).
No weights were modified; dtype {args.dtype}.

Key remap: `temb_mods.{{i}}.{{weight,bias}}` -> `model.layers.{{i}}.rms_norm_modulations.modulation.{{weight,bias}}`;
`insertion_head.*` and `time_mlp.*` unchanged; backbone keys unchanged.

Produced by `xlm-models/flexmdm/scripts/merge_flexmdm_dream_checkpoint.py`.
Paper: *From Interface to Inference* (FlexMDM); original license and credits belong to the authors.
"""
    )
    print(f"      files: {sorted(p.name for p in out.iterdir())}")

    # ---- optional push ----
    if args.push_to:
        from huggingface_hub import HfApi

        print(f"[5/5] pushing to {args.push_to} ...", flush=True)
        api = HfApi()
        api.create_repo(args.push_to, repo_type="model", private=args.private, exist_ok=True)
        api.upload_large_folder(repo_id=args.push_to, repo_type="model", folder_path=str(out))
        print(f"      done: https://huggingface.co/{args.push_to}")
    else:
        print("[5/5] skipped push (no --push-to)")


if __name__ == "__main__":
    main()
