# Copyright (c) 2023 Meta Platforms, Inc. and affiliates
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified from dplm/analysis/cal_plddt_dir.py.
# Original: https://github.com/bytedance/dplm/blob/main/analysis/cal_plddt_dir.py
#
# Changes (all additive, marked with "# PAPL-eval addition"):
#   - Extract pTM and mean pAE from ESMFold output tensors.
#   - Write a per-FASTA metrics.jsonl alongside the PDB output directory,
#     one JSON line per sequence: {header, length, plddt, ptm, pae, foldable}.
#     foldable = plddt > 80 AND ptm > 0.7 AND pae < 10  (PAPL Table 1 threshold).
#   - All original logic (batching, OOM handling, FASTA reading, PDB naming) unchanged.

import argparse
import glob
import json  # PAPL-eval addition
import logging
import os
import re
import sys
import typing as T
from pathlib import Path
from timeit import default_timer as timer

import esm
import torch

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


PathLike = T.Union[str, Path]


def read_fasta(
    path,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    with open(path, "r") as f:
        for result in read_alignment_lines(
            f,
            keep_gaps=keep_gaps,
            keep_insertions=keep_insertions,
            to_upper=to_upper,
        ):
            yield result


def read_alignment_lines(
    lines,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    seq = desc = None

    def parse(s):
        if not keep_gaps:
            s = re.sub("-", "", s)
        if not keep_insertions:
            s = re.sub("[a-z]", "", s)
        return s.upper() if to_upper else s

    for line in lines:
        if len(line) > 0 and line[0] == ">":
            if seq is not None and "X" not in seq:
                yield desc, parse(seq)
            desc = line.strip().lstrip(">")
            seq = ""
        else:
            assert isinstance(seq, str)
            seq += line.strip()
    assert isinstance(seq, str) and isinstance(desc, str)
    if "X" not in seq:
        yield desc, parse(seq)


def enable_cpu_offloading(model):
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import enable_wrap, wrap

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:9999",
        world_size=1,
        rank=0,
    )

    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model


def init_model_on_gpu_with_cpu_offloading(model):
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model


def create_batched_sequence_datasest(
    sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:

    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--fasta_dir",
        help="Path to directory containing input FASTA files (searched recursively)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--pdb",
        help="Output directory for PDB files and metrics.jsonl",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        help="Parent path to pretrained ESM data directory",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Number of recycles (default: 4, as used during ESMFold training).",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help=(
            "Maximum residue-tokens per GPU forward pass. Shorter sequences are "
            "grouped together for efficiency. Lower this (e.g. to 512) for "
            "sequences ≥ 600aa or if you see CUDA OOM errors."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Chunk axial attention to reduce peak memory from O(L²) to O(L). "
            "Recommended values: 128, 64, 32. Slower but more memory-efficient."
        ),
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    return parser


def run(args):
    logger.info("Loading ESMFold model")

    if args.model_dir is not None:
        torch.hub.set_dir(args.model_dir)

    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    model.set_chunk_size(args.chunk_size)

    if args.cpu_only:
        model.esm.float()
        model.cpu()
    elif args.cpu_offload:
        model = init_model_on_gpu_with_cpu_offloading(model)
    else:
        model.cuda()

    for fasta in glob.glob(f"{args.fasta_dir}/**/*.fasta", recursive=True):
        print(fasta)
        if args.pdb is not None:
            pdbdir = os.path.join(args.pdb, os.path.basename(fasta)[:-6])
        else:
            pdbdir = os.path.join(os.path.dirname(fasta), "esmfold_pdb")
        Path(pdbdir).mkdir(parents=True, exist_ok=True)

        # PAPL-eval addition: one JSONL file per input FASTA, co-located with PDBs.
        # Truncate on each run — the script processes the full FASTA from scratch.
        metrics_path = Path(pdbdir) / "metrics.jsonl"
        metrics_fh = metrics_path.open("w")

        logger.info(f"Reading sequences from {fasta}")
        all_sequences = sorted(
            read_fasta(fasta), key=lambda header_seq: len(header_seq[1])
        )
        logger.info(f"Loaded {len(all_sequences)} sequences")
        logger.info("Starting predictions")
        batched_sequences = create_batched_sequence_datasest(
            all_sequences, args.max_tokens_per_batch
        )

        num_completed = 0
        num_sequences = len(all_sequences)
        for headers, sequences in batched_sequences:
            start = timer()
            try:
                output = model.infer(sequences, num_recycles=args.num_recycles)
            except RuntimeError as e:
                if e.args[0].startswith("CUDA out of memory"):
                    if len(sequences) > 1:
                        logger.info(
                            f"CUDA OOM on batch of {len(sequences)} sequences. "
                            "Try lowering --max-tokens-per-batch."
                        )
                    else:
                        logger.info(
                            f"CUDA OOM on sequence {headers[0]} "
                            f"(length {len(sequences[0])})."
                        )
                    continue
                raise

            output = {key: value.cpu() for key, value in output.items()}
            pdbs = model.output_to_pdb(output)
            tottime = timer() - start
            time_string = f"{tottime / len(headers):0.1f}s"
            if len(sequences) > 1:
                time_string += f" (amortized, batch size {len(sequences)})"

            # PAPL-eval addition: extract pAE — shape (batch, padded_L, padded_L).
            # Crop to actual sequence length before averaging to avoid padding.
            pae_full = output["predicted_aligned_error"]

            for i, (header, seq, pdb_string, mean_plddt, ptm) in enumerate(zip(
                headers,
                sequences,
                pdbs,
                output["mean_plddt"],
                output["ptm"],
            )):
                output_file = Path(pdbdir) / f"{header}_plddt_{mean_plddt}.pdb"
                output_file.write_text(pdb_string)
                num_completed += 1

                # PAPL-eval addition: write metrics record.
                plddt_f = float(mean_plddt)
                ptm_f = float(ptm)
                sl = len(seq)
                pae_f = float(pae_full[i, :sl, :sl].mean())
                foldable = plddt_f > 80.0 and ptm_f > 0.7 and pae_f < 10.0
                metrics_fh.write(
                    json.dumps({
                        "header": header,
                        "length": len(seq),
                        "plddt": round(plddt_f, 4),
                        "ptm": round(ptm_f, 4),
                        "pae": round(pae_f, 4),
                        "foldable": foldable,
                    }) + "\n"
                )
                metrics_fh.flush()

                logger.info(
                    f"Predicted structure for {header}  len={len(seq)}"
                    f"  pLDDT={plddt_f:0.1f}  pTM={ptm_f:0.3f}  pAE={pae_f:0.2f}"
                    f"  foldable={'Y' if foldable else 'N'}"
                    f"  {time_string}"
                    f"  [{num_completed}/{num_sequences}]"
                )

        metrics_fh.close()
        logger.info(f"Metrics written to {metrics_path}")


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
