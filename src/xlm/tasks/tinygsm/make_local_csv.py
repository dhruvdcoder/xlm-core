"""Write a small streamed sample of TinyGSM/TinyGSM to a local CSV.

Streaming avoids downloading the full train split. The CSV has question and
code columns and is consumed by the tinygsm_local_sample dataset configs
(LocalDatasetManager, ds_type csv).

Usage:
    DATA_DIR=/tmp/xlm_data python -m xlm.tasks.tinygsm.make_local_csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
from pathlib import Path
from typing import Iterable, Mapping


def write_tinygsm_sample_csv(
    out_path: Path,
    num_examples: int = 10,
    split: str = "train",
    repo_id: str = "TinyGSM/TinyGSM",
) -> Path:
    """Stream the first num_examples rows of a TinyGSM split into a CSV.

    quoting is QUOTE_ALL so multi-line code survives the Hugging Face csv
    loader.
    """
    if num_examples < 1:
        raise ValueError("num_examples must be at least 1")

    from datasets import load_dataset

    stream: Iterable[Mapping[str, str]] = load_dataset(
        repo_id, split=split, streaming=True
    )
    rows = list(itertools.islice(stream, num_examples))
    if len(rows) != num_examples:
        raise RuntimeError(
            f"TinyGSM split {split!r} yielded {len(rows)} rows, "
            f"expected {num_examples}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "code"],
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"question": row["question"], "code": row["code"]}
            )
    return out_path


def main() -> None:
    default_dir = Path(os.environ.get("DATA_DIR", "data")) / "tinygsm_sample"
    parser = argparse.ArgumentParser(
        description=(
            "Stream the first N TinyGSM examples into a local CSV "
            "without downloading the full dataset."
        )
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of rows to stream (default: 10).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_dir,
        help=f"Output directory (default: {default_dir}).",
    )
    parser.add_argument(
        "--file-name",
        default="train.csv",
        help="CSV file name inside --out-dir (default: train.csv).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Hugging Face split to stream (default: train).",
    )
    args = parser.parse_args()

    out_path = write_tinygsm_sample_csv(
        args.out_dir / args.file_name,
        num_examples=args.num_examples,
        split=args.split,
    )
    print(f"Wrote {args.num_examples} TinyGSM rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
