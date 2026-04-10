#!/usr/bin/env python3
"""Aggregate per-sequence ESMFold metrics from a completed eval run.

Reads:
  - FASTA files under --fasta-dir  (for sequences, diversity, AA entropy)
  - metrics.jsonl files under --pdb-root/<stem>/  (for pLDDT, pTM, pAE,
    foldability; written by cal_plddt_papl.py)

Reports:
  - Per-length breakdown: foldability %, mean pLDDT, mean pTM, mean pAE
  - Corpus-level: same averages pooled across all lengths
  - Diversity: % unique sequences across all FASTA files
  - Pooled 20-AA Shannon entropy (nats)

This mirrors the structure of PAPL Table 1. Diversity and entropy are
always available (computed from FASTA alone); the structural metrics
require a completed ESMFold run.

Usage:
  python evals/protein_eval/aggregate_metrics.py \\
    --fasta-dir evals/protein_eval/<exp>/by_length \\
    --pdb-root  evals/protein_eval/<exp>/esmfold_pdb
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


# ── FASTA parsing ─────────────────────────────────────────────────────────────

def read_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    desc: Optional[str] = None
    chunks: List[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == ">":
                if desc is not None:
                    yield desc, "".join(chunks)
                desc = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if desc is not None:
        yield desc, "".join(chunks)


# ── Metrics JSONL ─────────────────────────────────────────────────────────────

def load_metrics_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Diversity / entropy helpers ───────────────────────────────────────────────

def shannon_nats(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c:
            p = c / total
            h -= p * math.log(p)
    return h


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt(val: Optional[float], fmt: str = ".2f") -> str:
    return f"{val:{fmt}}" if val is not None else "n/a"


def _mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fasta-dir",
        type=Path,
        default=here / "example_data" / "by_length",
        help="Directory containing .fasta files grouped by length.",
    )
    p.add_argument(
        "--pdb-root",
        type=Path,
        required=False,
        default=None,
        help=(
            "Root output directory from run_esmfold.sh (contains one subdirectory "
            "per FASTA file, each holding PDB files and a metrics.jsonl). "
            "If omitted, only sequence-level statistics are reported."
        ),
    )
    args = p.parse_args()

    fasta_paths = sorted(Path(args.fasta_dir).glob("*.fasta"))
    if not fasta_paths:
        print(f"No .fasta files found in {args.fasta_dir}")
        return 1

    all_seqs: List[str] = []
    aa_counts: Counter = Counter()

    # Per-length structural metrics collected from jsonl.
    # Key: length bucket (int), value: list of metric dicts.
    by_length: Dict[int, List[dict]] = defaultdict(list)
    all_records: List[dict] = []
    missing_jsonl: List[str] = []

    for fasta_path in fasta_paths:
        stem = fasta_path.stem
        print(f"\n{'─'*60}")
        print(f"FASTA : {fasta_path.name}")

        seqs_this_file: List[str] = []
        for _header, seq in read_fasta(fasta_path):
            all_seqs.append(seq)
            seqs_this_file.append(seq)
            aa_counts.update(seq.upper())

        print(f"  Sequences : {len(seqs_this_file)}")

        if args.pdb_root is not None:
            jsonl = Path(args.pdb_root) / stem / "metrics.jsonl"
            if jsonl.is_file():
                records = load_metrics_jsonl(jsonl)
                print(f"  metrics.jsonl : {len(records)} records  ({jsonl})")
                for rec in records:
                    length = rec.get("length", 0)
                    by_length[length].append(rec)
                    all_records.append(rec)
            else:
                print(f"  metrics.jsonl : NOT FOUND ({jsonl})")
                missing_jsonl.append(stem)

    # ── Per-length table ──────────────────────────────────────────────────────
    if all_records:
        print(f"\n{'═'*72}")
        print(f"{'Length':>8}  {'N':>5}  {'Foldable%':>10}  {'pLDDT':>7}  {'pTM':>6}  {'pAE':>6}")
        print(f"{'─'*8}  {'─'*5}  {'─'*10}  {'─'*7}  {'─'*6}  {'─'*6}")
        for length in sorted(by_length):
            recs = by_length[length]
            n = len(recs)
            n_fold = sum(1 for r in recs if r.get("foldable", False))
            fold_pct = 100.0 * n_fold / n if n else 0.0
            mean_plddt = _mean([r["plddt"] for r in recs if "plddt" in r])
            mean_ptm = _mean([r["ptm"] for r in recs if "ptm" in r])
            mean_pae = _mean([r["pae"] for r in recs if "pae" in r])
            print(
                f"{length:>8}  {n:>5}  {fold_pct:>9.1f}%"
                f"  {_fmt(mean_plddt):>7}  {_fmt(mean_ptm):>6}  {_fmt(mean_pae):>6}"
            )

        # Corpus totals
        n_all = len(all_records)
        n_fold_all = sum(1 for r in all_records if r.get("foldable", False))
        fold_pct_all = 100.0 * n_fold_all / n_all
        m_plddt = _mean([r["plddt"] for r in all_records if "plddt" in r])
        m_ptm = _mean([r["ptm"] for r in all_records if "ptm" in r])
        m_pae = _mean([r["pae"] for r in all_records if "pae" in r])
        print(f"{'─'*8}  {'─'*5}  {'─'*10}  {'─'*7}  {'─'*6}  {'─'*6}")
        print(
            f"{'ALL':>8}  {n_all:>5}  {fold_pct_all:>9.1f}%"
            f"  {_fmt(m_plddt):>7}  {_fmt(m_ptm):>6}  {_fmt(m_pae):>6}"
        )
        print(f"{'═'*72}")
        print(
            "Foldable = pLDDT > 80 AND pTM > 0.7 AND pAE < 10  (PAPL threshold)"
        )

    # ── Sequence-level stats ──────────────────────────────────────────────────
    print()
    n_total = len(all_seqs)
    n_unique = len(set(all_seqs))
    diversity_pct = 100.0 * n_unique / n_total if n_total else 0.0
    print(f"Total sequences : {n_total}")
    print(f"Unique sequences: {n_unique}  ({diversity_pct:.2f}% diversity)")

    standard = "ACDEFGHIKLMNPQRSTVWY"
    aa20 = Counter({k: aa_counts.get(k, 0) for k in standard})
    entropy = shannon_nats(aa20)
    print(f"AA entropy (20 standard AA, nats): {entropy:.4f}  [max = {math.log(20):.4f}]")

    if missing_jsonl:
        print(f"\nWARNING: metrics.jsonl missing for {len(missing_jsonl)} FASTA(s): {missing_jsonl}")
        print("  Structural metrics above are partial. Re-run run_esmfold.sh for missing files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
