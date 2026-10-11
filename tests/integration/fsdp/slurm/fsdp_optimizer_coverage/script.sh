#!/bin/bash
# SLURM job: 2-GPU FSDP optimizer-coverage integration test.
#
# Invoked by ``tests.integration._slurm.submit_sbatch_and_wait`` as::
#
#     sbatch --wait <this script> <result_dir> [config_json]
#
# Positional arguments
# --------------------
# $1 : result_dir   -- absolute path; ``rank_<RANK>.json`` files, the
#                      Hydra run directory (logs/) and the slurm log land here.
# $2 : config_json  -- (optional) JSON blob forwarded to script.py via
#                      ``--config-json`` (harness class, use_orig_params, ...).
#
# Resource directives default to a small 2-GPU job. Override partition /
# constraint / QoS on the sbatch command line via ``extra_sbatch_args``
# (``XLM_INTEGRATION_SBATCH_ARGS`` in the pytest wrapper), e.g. on Unity:
#   XLM_INTEGRATION_SBATCH_ARGS="--partition=gpu,--constraint=a100|l40s|v100"
#
#SBATCH --job-name=xlm_fsdp_cov
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <result_dir> [config_json]" >&2
    exit 64
fi

RESULT_DIR="$1"
# NB: "${2:-{}}" would parse as "${2:-{}" + "}" and append a stray brace.
CONFIG_JSON="${2:-}"
if [[ -z "${CONFIG_JSON}" ]]; then
    CONFIG_JSON="{}"
fi
mkdir -p "${RESULT_DIR}"

# Resolve the repo root. sbatch copies this script into the slurmd spool, so
# BASH_SOURCE does not point into the repo; the pytest wrapper exports
# XLM_REPO_ROOT (sbatch forwards the environment). Fall back to the submit
# directory, which is the repo root when pytest runs from there.
if [[ -n "${XLM_REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd -- "${XLM_REPO_ROOT}" && pwd)"
else
    REPO_ROOT="$(cd -- "${SLURM_SUBMIT_DIR:-.}" && pwd)"
fi
SCRIPT_DIR="${REPO_ROOT}/tests/integration/fsdp/slurm/fsdp_optimizer_coverage"
if [[ ! -f "${SCRIPT_DIR}/script.py" ]]; then
    echo "cannot locate ${SCRIPT_DIR}/script.py; set XLM_REPO_ROOT" >&2
    exit 65
fi

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PROJECT_ROOT="${REPO_ROOT}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

# Copy the slurm log next to the per-rank result files on exit.
trap 'cp -f "${SLURM_SUBMIT_DIR:-.}/xlm_fsdp_cov_${SLURM_JOB_ID}.out" "${RESULT_DIR}/" 2>/dev/null || true' EXIT

# 1. Prepare the star_easy data once (single process, same overlay as the
#    CPU smoke tests). Reuses DATA_DIR / HF_HOME caches when set.
cd "${REPO_ROOT}"
python -m xlm \
    --config-dir "${REPO_ROOT}/tests/configs" \
    job_type=prepare_data \
    job_name=prepare_star_easy_mlm \
    experiment=star_easy_mlm \
    debug=smoke \
    trainer_strategy=cpu \
    "paths.log_dir=${RESULT_DIR}/prepare_logs"

# 2. One process per GPU. Lightning's SLURMEnvironment picks up
#    SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID; MASTER_ADDR / MASTER_PORT
#    are exported for torch.distributed.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=$((10000 + (SLURM_JOB_ID % 50000)))

srun --export=ALL,MASTER_ADDR,MASTER_PORT \
    python -u "${SCRIPT_DIR}/script.py" \
        --result-dir "${RESULT_DIR}" \
        --config-json "${CONFIG_JSON}"
