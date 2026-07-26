#!/usr/bin/env python3
"""Convert Fast-dLLM c40m60 conversation shards → plain prompt/answer parquet for LLaDA.

Reads ``train_conversation/train_*.json`` (messages + source), strips to plain
text (no ChatML), writes a single parquet plus a small manifest. Tokenization
happens later in ``mix_preprocess_fn`` with the LLaDA tokenizer.

Example::

    python xlm-models/llada/scripts/convert_c40m60_for_llada.py \\
      --in_dir /path/to/opencode_openmath_60k_c40m60 \\
      --out_dir /path/to/opencode_openmath_60k_c40m60_llada
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _iter_instances(in_dir: Path):
    conv = in_dir / "train_conversation"
    if not conv.is_dir():
        raise FileNotFoundError(f"missing train_conversation under {in_dir}")
    shards = sorted(conv.glob("train_*.json"))
    if not shards:
        raise FileNotFoundError(f"no train_*.json under {conv}")
    for shard in shards:
        with shard.open() as f:
            payload = json.load(f)
        for inst in payload.get("instances") or []:
            yield inst


def _to_row(inst: dict) -> dict | None:
    messages = inst.get("messages") or []
    user = next((m.get("content") for m in messages if m.get("role") == "user"), None)
    asst = next(
        (m.get("content") for m in messages if m.get("role") == "assistant"), None
    )
    if user is None or asst is None:
        return None
    prompt = str(user).strip()
    answer = str(asst).strip()
    if not prompt or not answer:
        return None
    # Guardrail: never emit ChatML markup into LLaDA training rows.
    for bad in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        if bad in prompt or bad in answer:
            raise ValueError(f"ChatML/special markup found in row source={inst.get('source')}")
    source = str(inst.get("source") or "unknown")
    return {"prompt": prompt, "answer": answer, "source": source}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    skipped = 0
    sources: Counter[str] = Counter()
    for inst in _iter_instances(args.in_dir):
        row = _to_row(inst)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        sources[row["source"]] += 1

    if not rows:
        print("ERROR: no rows converted", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    out_parquet = args.out_dir / "train.parquet"
    pq.write_table(table, out_parquet)

    manifest = {
        "n_rows": len(rows),
        "n_skipped": skipped,
        "sources": dict(sources),
        "in_dir": str(args.in_dir.resolve()),
        "out_parquet": str(out_parquet.resolve()),
        "note": "plain prompt/answer; no ChatML; tokenize with LLaDA at train time",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
