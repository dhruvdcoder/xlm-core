#!/bin/bash
# xlm-core FlexMDM (Dream-Coder-7B) fine-tune on Unity with FSDP; the counterpart of
# unity_flexmdm_ref_train.sbatch (same data manifest, seed, batch structure, LR schedule, GPUs).
#
#   sbatch --gres=gpu:4 xlm-models/flexmdm/scripts/unity_flexmdm_dream_train.sbatch
#   STEPS=1000 GLOBAL_BATCH=288 sbatch --gres=gpu:4 ...
#SBATCH --job-name=flexmdm_dream_train
#SBATCH --partition=superpod-a100,gpu
#SBATCH --constraint=vram80,bf16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail

WORKTREE=${WORKTREE:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core-dream-flexmdm}
VENV=${VENV:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core/.venv_xlm_core}
DATA_ROOT=${DATA_ROOT:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/flexmdm_data}
export FLEXMDM_PRETOKENIZED_ROOT=${FLEXMDM_PRETOKENIZED_ROOT:-$DATA_ROOT/pretokenized}
STEPS=${STEPS:-500}
GLOBAL_BATCH=${GLOBAL_BATCH:-576}
MICRO=${MICRO:-6}
SEED=${SEED:-1}
NPROC=${SLURM_NTASKS_PER_NODE:-4}
RUN_NAME=${RUN_NAME:-xlm_fsdp_s${STEPS}_gb${GLOBAL_BATCH}_g${NPROC}_seed${SEED}}

# Lmod is not initialized in non-login batch shells on some Unity partitions.
if ! command -v module >/dev/null 2>&1; then
  for s in /etc/profile.d/z0[0-9]-lmod-*.sh; do [ -r "$s" ] && source "$s"; done
fi
module load conda/latest
conda activate "$VENV"
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE/src:${PYTHONPATH:-}"   # worktree src wins over the editable install
set -a; [ -f .env ] && source .env; [ -f .secrets.env ] && source .secrets.env; set +a
export HF_TOKEN=${HF_TOKEN:-${HF_HUB_KEY:-}}
export WANDB_PROJECT=${WANDB_PROJECT_OVERRIDE:-flexmdm-dream-repro}
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
set -u
[ -f "$FLEXMDM_PRETOKENIZED_ROOT/manifest.json" ] || { echo "missing $FLEXMDM_PRETOKENIZED_ROOT/manifest.json"; exit 2; }

echo "=== xlm FlexMDM-Dream train: steps=$STEPS global_batch=$GLOBAL_BATCH micro=$MICRO gpus=$NPROC seed=$SEED run=$RUN_NAME"
srun --kill-on-bad-exit=1 xlm job_type=train loggers=wandb \
    experiment=[code_sft_flexmdm_dream,flexmdm_dream_fsdp_args] \
    trainer_strategy=fsdp trainer.devices="$NPROC" trainer.num_nodes=1 \
    job_name="$RUN_NAME" seed="$SEED" \
    per_device_batch_size="$MICRO" global_batch_size="$GLOBAL_BATCH" \
    trainer.max_steps="$STEPS" \
    lr_scheduler.num_training_steps="$STEPS" lr_scheduler.num_warmup_steps=$((STEPS / 10)) \
    trainer.val_check_interval="$STEPS" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps="$STEPS" \
    num_dataloader_workers=4 \
    "$@"
