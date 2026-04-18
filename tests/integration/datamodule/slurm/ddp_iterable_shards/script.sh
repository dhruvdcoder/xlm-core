#!/bin/bash
# SLURM job: 2-rank DDP integration test for ``DatasetManager``.
#
# Invoked by ``tests.integration._slurm.submit_sbatch_and_wait`` as::
#
#     sbatch --wait <this script> <result_dir> [config_json]
#
# Positional arguments
# --------------------
# $1 : result_dir   -- absolute path; ``rank_<RANK>.json`` files are
#                      written here, plus the ``slurm-*.out`` log.
# $2 : config_json  -- (optional) JSON blob forwarded to script.py via
#                      ``--config-json``.
#
# The SLURM resource directives default to a small job that fits in the
# common ``gpu`` partition.  Override them on the ``sbatch`` command
# line via ``extra_sbatch_args`` in the test if you need to target a
# specific partition / QoS.
#
#SBATCH --job-name=xlm_dsm_int
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <result_dir> [config_json]" >&2
    exit 64
fi

RESULT_DIR="$1"
CONFIG_JSON="${2:-{}}"

# Per-job log goes next to the per-rank result files so users can
# inspect everything in one place.
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Resolve the workspace root from this script's location:
# tests/integration/datamodule/slurm/ddp_iterable_shards/script.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." &> /dev/null && pwd)"

# Make ``xlm.*`` and ``tests.integration.*`` importable inside srun.
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

# Avoid runaway BLAS thread fan-out and HF progress-bar noise in logs.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

# torch.distributed needs MASTER_ADDR / MASTER_PORT.  Use the first
# allocated host and a port derived from the job id to avoid clashes
# with concurrent jobs on the same node.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=$((10000 + (SLURM_JOB_ID % 50000)))

# srun sets RANK / WORLD_SIZE / LOCAL_RANK on each task, but only when
# ``--mpi=pmix``-style integration is enabled.  Export the equivalent
# values from SLURM_* so script.py sees the same env on every cluster.
srun --export=ALL,MASTER_ADDR,MASTER_PORT \
    bash -c '
        export RANK=${SLURM_PROCID}
        export WORLD_SIZE=${SLURM_NTASKS}
        export LOCAL_RANK=${SLURM_LOCALID}
        exec python -u "'"${SCRIPT_DIR}"'/script.py" \
            --result-dir "'"${RESULT_DIR}"'" \
            --config-json '"'""${CONFIG_JSON}""'"'
    '
