#!/bin/bash
# HumanEval / HumanEval+ pass@k for FlexMDM (Dream-Coder-7B) on Unity, 1x 80GB GPU.
# Paper setting: 164 tasks x 16 samples, 512 steps, T=0.1, top-k confidence, T_ins=1.0, L=768.
# Reported pass@1: HE 50.65, HE+ 46.61 (n=16 estimator).
#
# Submit from the worktree root:
#   sbatch xlm-models/flexmdm/scripts/unity_flexmdm_dream_humaneval.sbatch
# Env overrides: N_SAMPLES (16), LIMIT (null), BATCH (16), STEPS (512), JOB (job_name).
#SBATCH --job-name            flexmdm_dream_he
#SBATCH --nodes               1
#SBATCH --ntasks-per-node     1
#SBATCH --gres                gpu:1
#SBATCH --cpus-per-task       8
#SBATCH --mem                 64GB
#SBATCH --time                1-00:00:00
#SBATCH --partition           superpod-a100,gpu,gpu-preempt
#SBATCH --constraint          vram80,bf16
#SBATCH --open-mode           append
#SBATCH --output              logs/%x-%j.out
#SBATCH --mail-type           BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user           dhruveshpate@umass.edu

set -euo pipefail
WORKTREE=${WORKTREE:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core-dream-flexmdm}
VENV=${VENV:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core/.venv_xlm_core}

# Lmod is not initialized in non-login batch shells on some Unity partitions.
if ! command -v module >/dev/null 2>&1; then
  for s in /etc/profile.d/z0[0-9]-lmod-*.sh; do [ -r "$s" ] && source "$s"; done
fi
module load conda/latest
conda activate "$VENV"
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE/src:${PYTHONPATH:-}"   # worktree src wins over the editable install
set -a; [ -f .env ] && source .env; [ -f .secrets.env ] && source .secrets.env; set +a
export HF_TOKEN="${HF_TOKEN:-${HF_HUB_KEY:-}}"      # private hub repo (tokenizer + weights)
export TOKENIZERS_PARALLELISM=false

nvidia-smi -L
echo "commit: $(git rev-parse --short HEAD)  worktree: $WORKTREE"

N_SAMPLES=${N_SAMPLES:-16}
LIMIT=${LIMIT:-null}
BATCH=${BATCH:-16}
STEPS=${STEPS:-512}
JOB=${JOB:-flexmdm_dream_he_n${N_SAMPLES}_s${STEPS}}

srun xlm job_type=eval \
  experiment=humaneval_flexmdm_dream_eval \
  job_name="$JOB" \
  trainer_strategy=single_device \
  humaneval_n_samples="$N_SAMPLES" humaneval_limit="$LIMIT" \
  per_device_batch_size="$BATCH" global_batch_size="$BATCH" \
  predictor.max_steps="$STEPS" \
  post_hoc_evaluator.evaluators.humaneval.workers="$SLURM_CPUS_PER_TASK" \
  "$@"
