"""Histogram of CharBPE token lengths for IWSLT14 DE→EN (RDM text v1).

Reports prompt (source), output (target), and joint (prompt + output) lengths
from a tokenized HF dataset cache (columns ``prompt_token_ids`` /
``input_token_ids``).

Example::

    python -m xlm.tasks.iwslt14_de_en.analyze_token_lengths \\
        --dataset-root /path/to/iwslt14_de_en_rdm_text_v1_tok_... \\
        --split train \\
        --threshold 128 \\
        --json-out /tmp/iwslt_lengths.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from datasets import load_from_disk


DEFAULT_ROOT = (
    "/scratch4/workspace/dhruveshpate_umass_edu-learned-noise/"
    "learned-noise/data/local/"
    "iwslt14_de_en_rdm_text_v1_tok_975aedcb168b_rawref"
)


def _histogram(
    values: np.ndarray, bin_width: int, max_edge: int | None = None
) -> Dict[str, Any]:
    """Fixed-width histogram; last bin is open-ended if values exceed max_edge."""
    vmax = int(values.max()) if len(values) else 0
    if max_edge is None:
        max_edge = int(math.ceil((vmax + 1) / bin_width) * bin_width)
    edges = list(range(0, max_edge + bin_width, bin_width))
    # Catch overflow in a final [max_edge, inf) bin.
    counts, _ = np.histogram(values, bins=edges + [np.inf])
    categories: List[str] = []
    for i in range(len(edges) - 1):
        categories.append(f"{edges[i]}-{edges[i + 1] - 1}")
    categories.append(f"{max_edge}+")
    return {
        "bin_width": bin_width,
        "categories": categories,
        "counts": [int(c) for c in counts.tolist()],
        "edges": edges,
    }


def _summarize(
    values: np.ndarray, thresholds: Sequence[int] = (128,)
) -> Dict[str, Any]:
    if len(values) == 0:
        return {"n": 0}
    out: Dict[str, Any] = {
        "n": int(len(values)),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }
    for threshold in thresholds:
        out[f"gt_{threshold}"] = int((values > threshold).sum())
        out[f"gt_{threshold}_frac"] = float((values > threshold).mean())
        out[f"gte_{threshold}"] = int((values >= threshold).sum())
        out[f"gte_{threshold}_frac"] = float((values >= threshold).mean())
    return out


def analyze_split(
    dataset_root: Path,
    split: str,
    threshold: int = 128,
    bin_width: int = 16,
    max_edge: int = 256,
) -> Dict[str, Any]:
    ds = load_from_disk(str(dataset_root / split))
    prompt = np.fromiter(
        (len(x) for x in ds["prompt_token_ids"]), dtype=np.int32, count=len(ds)
    )
    output = np.fromiter(
        (len(x) for x in ds["input_token_ids"]), dtype=np.int32, count=len(ds)
    )
    # Collator concatenates prefix + BOS + (dropped) target; BOS adds +1.
    joint = prompt + output
    joint_with_bos = prompt + output + 1

    side_thr = threshold
    joint_thr = threshold * 2
    return {
        "dataset_root": str(dataset_root),
        "split": split,
        "n_examples": len(ds),
        "threshold": threshold,
        "lengths": {
            "prompt": _summarize(prompt, (side_thr,)),
            "output": _summarize(output, (side_thr,)),
            "joint": _summarize(joint, (side_thr, joint_thr)),
            "joint_with_bos": _summarize(
                joint_with_bos, (side_thr, joint_thr)
            ),
        },
        "histograms": {
            "prompt": _histogram(prompt, bin_width, max_edge),
            "output": _histogram(output, bin_width, max_edge),
            "joint": _histogram(joint, bin_width, max_edge * 2),
            "joint_with_bos": _histogram(
                joint_with_bos, bin_width, max_edge * 2
            ),
        },
        "overflow": {
            "prompt_gt_block": int((prompt > side_thr).sum()),
            "output_gt_block": int((output > side_thr).sum()),
            "either_side_gt_block": int(
                ((prompt > side_thr) | (output > side_thr)).sum()
            ),
            "joint_gt_concat": int((joint > joint_thr).sum()),
        },
        "notes": {
            "prompt": "len(prompt_token_ids) = source CharBPE length (no specials)",
            "output": "len(input_token_ids) = target CharBPE length (no specials)",
            "joint": "prompt + output (raw token counts, no specials)",
            "joint_with_bos": (
                "prompt + output + 1 (BOS); closer to ILM concat before drops/pad"
            ),
            "threshold": (
                f"ILM experiment uses input_block_size=block_size={threshold}; "
                f"concat padded length is {joint_thr}"
            ),
        },
    }


def _print_summary(report: Dict[str, Any]) -> None:
    thr = report["threshold"]
    print(f"split={report['split']}  n={report['n_examples']}  threshold={thr}")
    for name, stats in report["lengths"].items():
        extras = []
        for key in sorted(k for k in stats if k.startswith("gt_")):
            if key.endswith("_frac"):
                continue
            t = key.split("_", 1)[1]
            frac = stats[f"gt_{t}_frac"]
            extras.append(f">{t}={stats[key]} ({100 * frac:.2f}%)")
        print(
            f"  {name:16s}  "
            f"min={stats['min']:4d}  max={stats['max']:4d}  "
            f"mean={stats['mean']:6.1f}  p95={stats['p95']:6.1f}  "
            f"p99={stats['p99']:6.1f}  " + "  ".join(extras)
        )
    ov = report["overflow"]
    print(
        f"  overflow vs block={thr}: "
        f"prompt={ov['prompt_gt_block']}  output={ov['output_gt_block']}  "
        f"either={ov['either_side_gt_block']}  "
        f"joint>{thr * 2}={ov['joint_gt_concat']}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(DEFAULT_ROOT),
        help="Tokenized HF dataset root containing train/validation/test",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "validation", "test", "all"),
    )
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--bin-width", type=int, default=16)
    parser.add_argument("--max-edge", type=int, default=256)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full JSON report",
    )
    args = parser.parse_args(argv)

    splits = (
        ["train", "validation", "test"]
        if args.split == "all"
        else [args.split]
    )
    reports = [
        analyze_split(
            args.dataset_root,
            split,
            threshold=args.threshold,
            bin_width=args.bin_width,
            max_edge=args.max_edge,
        )
        for split in splits
    ]

    for report in reports:
        _print_summary(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload: Any = reports[0] if len(reports) == 1 else reports
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
