#!/usr/bin/env bash
# Run ESMFold structure prediction on a directory tree of FASTA files.
#
# Activates the eval venv, sets PYTHONPATH for vendored OpenFold, and
# calls cal_plddt_papl.py. All three confidence scalars (pLDDT, pTM, pAE)
# plus the foldability flag are written to a metrics.jsonl alongside each
# group of PDB files.
#
# Usage:
#   FASTA_DIR=evals/protein_eval/<exp>/by_length \
#   PDB_DIR=evals/protein_eval/<exp>/esmfold_pdb \
#   bash evals/protein_eval/run_esmfold.sh
#
# Optional overrides:
#   DPLM_ROOT   — path to dplm repo (default: repo_root/vendor/dplm)
#   VENV        — path to eval venv  (default: DPLM_ROOT/.venv-esmfold)
#   MAX_TOKENS  — max tokens per GPU batch (default: 1024; lower to 512 for
#                 sequences ≥ 600aa or if you see CUDA OOM errors)
#
# Requires: eval venv built with setup_eval_venv.sh, run on a GPU node.
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
EVAL_DIR="${REPO_ROOT}/evals/protein_eval"
DPLM_ROOT="${DPLM_ROOT:-${REPO_ROOT}/vendor/dplm}"
VENV="${VENV:-${DPLM_ROOT}/.venv-esmfold}"

if [ -z "${FASTA_DIR:-}" ] || [ -z "${PDB_DIR:-}" ]; then
    echo "Usage: FASTA_DIR=... PDB_DIR=... bash $0"
    exit 1
fi

if [ ! -f "${VENV}/bin/activate" ]; then
    echo "ERROR: venv not found at ${VENV}"
    echo "  Run: bash ${EVAL_DIR}/setup_eval_venv.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${DPLM_ROOT}/vendor/openfold${PYTHONPATH:+:${PYTHONPATH}}"

exec python "${EVAL_DIR}/cal_plddt_papl.py" \
    --fasta_dir "${FASTA_DIR}" \
    --pdb       "${PDB_DIR}" \
    --max-tokens-per-batch "${MAX_TOKENS:-1024}" \
    "$@"
