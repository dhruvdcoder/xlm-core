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
  case $- in *u*) _had_u=1;; *) _had_u=0;; esac; set +u
  for s in /etc/profile.d/z0[0-9]-lmod-*.sh; do [ -r "$s" ] && source "$s"; done
  [ "$_had_u" = 1 ] && set -u
fi
module load conda/latest
conda activate "$VENV"
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE/src:$WORKTREE/xlm-models:${PYTHONPATH:-}"   # worktree src + xlm-models win over the editable installs
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

# One batch (16 x 768 tokens x 512 steps) takes ~10 min on an A100, i.e. ~27 h for all 164 tasks
# on a single GPU. Shard whole tasks across an array: `sbatch --array=0-3 <this script>`.
# Each shard writes its own predictions JSONL and skips the in-job pass@k; score all shards with
#   python -m xlm.tasks.humaneval.score --n-samples 16 --expect-tasks 164 \
#       --output-dir logs/${JOB}/humaneval_passk \
#       --predictions "logs/${JOB}_shard*/runs/*/predictions/test/humaneval_prediction/*.jsonl"
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
SHARD_ARGS=()
if [ "$NUM_SHARDS" -gt 1 ]; then
  JOB="${JOB}_shard${SHARD}of${NUM_SHARDS}"
  # drop the in-job pass@k (it would see only this shard); CompositePostHocEvaluator with no
  # matching evaluator returns empty metrics.
  SHARD_ARGS=(humaneval_shard_index="$SHARD" humaneval_num_shards="$NUM_SHARDS" '~post_hoc_evaluator.evaluators.humaneval')
else
  SHARD_ARGS=(post_hoc_evaluator.evaluators.humaneval.workers="$SLURM_CPUS_PER_TASK")
fi

srun xlm job_type=eval \
  experiment=humaneval_flexmdm_dream_eval \
  job_name="$JOB" \
  trainer_strategy=single_device \
  humaneval_n_samples="$N_SAMPLES" humaneval_limit="$LIMIT" \
  per_device_batch_size="$BATCH" global_batch_size="$BATCH" \
  predictor.max_steps="$STEPS" \
  "${SHARD_ARGS[@]}" \
  "$@"
