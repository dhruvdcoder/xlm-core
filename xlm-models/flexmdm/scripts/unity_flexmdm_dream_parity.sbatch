#!/bin/bash
# 7B parity: xlm-core FlexMDMDreamModel/Predictor vs reference FlexMDM (forward + same-seed traces).
# Needs ~34 GB GPU memory (both models in bf16). Submit from the worktree root:
#   sbatch xlm-models/flexmdm/scripts/unity_flexmdm_dream_parity.sbatch
#SBATCH --job-name            flexmdm_dream_parity
#SBATCH --nodes               1
#SBATCH --ntasks-per-node     1
#SBATCH --gres                gpu:1
#SBATCH --cpus-per-task       8
#SBATCH --mem                 80GB
#SBATCH --time                04:00:00
#SBATCH --partition           superpod-a100,gpu,gpu-preempt
#SBATCH --constraint          vram80,bf16
#SBATCH --open-mode           append
#SBATCH --output              logs/%x-%j.out
#SBATCH --mail-type           END,FAIL
#SBATCH --mail-user           dhruveshpate@umass.edu

set -euo pipefail
WORKTREE=${WORKTREE:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core-dream-flexmdm}
VENV=${VENV:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core/.venv_xlm_core}
REF_ROOT=${REF_ROOT:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/flexmdm_ref/FlexMDM}

module load conda/latest
conda activate "$VENV"
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE/src:${PYTHONPATH:-}"
set -a; [ -f .env ] && source .env; [ -f .secrets.env ] && source .secrets.env; set +a
export HF_TOKEN="${HF_TOKEN:-${HF_HUB_KEY:-}}"
export FLEXMDM_REF_ROOT="$REF_ROOT"

nvidia-smi -L
echo "commit: $(git rev-parse --short HEAD)"
mkdir -p logs/parity
OUT=logs/parity/flexmdm_dream_parity_${SLURM_JOB_ID}.json

srun python xlm-models/flexmdm/scripts/flexmdm_dream_parity.py \
  --ours dhruveshpatel/dream_flexmdm --ref yuyuanchen0/flexmdm \
  --n-prompts 3 --n-masks 64 --steps "${STEPS:-512}" --trace-prompts 2 \
  --out "$OUT" "$@"
echo "wrote $OUT"
