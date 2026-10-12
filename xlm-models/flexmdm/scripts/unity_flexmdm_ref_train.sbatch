#!/bin/bash
# Reference FlexMDM trainer (genuine-any-order/FlexMDM flexmdm/train_fsdp.py) for a short run on
# Unity: the baseline curve that xlm-core's code_sft_flexmdm_dream experiment must track.
#
# Same knobs as our run (see unity_flexmdm_dream_train.sbatch): STEPS optimizer steps with the
# reference recipe (global batch 576, micro 6, lr 1e-5 / x2 insertion, warmup 10% + cosine over
# STEPS, clip 1.0, seed 1), sdpa attention (no flash-attn in the venv), one node x NPROC GPUs.
#
#   sbatch --gres=gpu:4 xlm-models/flexmdm/scripts/unity_flexmdm_ref_train.sbatch
#   STEPS=1000 GLOBAL_BATCH=288 sbatch --gres=gpu:4 ...
#SBATCH --job-name=flexmdm_ref_train
#SBATCH --partition=superpod-a100,gpu
#SBATCH --constraint=vram80,bf16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail

WORKTREE=${WORKTREE:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core-dream-flexmdm}
VENV=${VENV:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core/.venv_xlm_core}
REF_ROOT=${REF_ROOT:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/flexmdm_ref/FlexMDM}
DATA_ROOT=${DATA_ROOT:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/flexmdm_data}
export PRETOKENIZED_ROOT=${PRETOKENIZED_ROOT:-$DATA_ROOT/pretokenized}
export SCRATCH_ROOT=$DATA_ROOT
export BASE_MODEL=${BASE_MODEL:-Dream-org/Dream-Coder-v0-Base-7B}
STEPS=${STEPS:-500}
GLOBAL_BATCH=${GLOBAL_BATCH:-576}
MICRO=${MICRO:-6}
SEED=${SEED:-1}
NPROC=${NPROC:-$(echo "${SLURM_GPUS_ON_NODE:-4}" | tr -d '[:alpha:]()')}
RUN_NAME=${RUN_NAME:-ref_fsdp_s${STEPS}_gb${GLOBAL_BATCH}_g${NPROC}_seed${SEED}}
export SAVE_ROOT=${SAVE_ROOT:-$DATA_ROOT/ref_checkpoints/$RUN_NAME}
export WANDB_PROJECT=${WANDB_PROJECT_OVERRIDE:-flexmdm-dream-repro}

# Lmod is not initialized in non-login batch shells on some Unity partitions.
if ! command -v module >/dev/null 2>&1; then
  for s in /etc/profile.d/z0[0-9]-lmod-*.sh; do [ -r "$s" ] && source "$s"; done
fi
module load conda/latest
conda activate "$VENV"
cd "$WORKTREE"
set -a; [ -f .env ] && source .env; [ -f .secrets.env ] && source .secrets.env; set +a
export HF_TOKEN=${HF_TOKEN:-${HF_HUB_KEY:-}}
export WANDB_PROJECT=${WANDB_PROJECT_OVERRIDE:-flexmdm-dream-repro} # .env sets a different default
export WANDB_DIR=$DATA_ROOT/wandb
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=$(hostname) MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
set -u
mkdir -p "$SAVE_ROOT" "$WANDB_DIR"
[ -f "$PRETOKENIZED_ROOT/manifest.json" ] || { echo "missing $PRETOKENIZED_ROOT/manifest.json (run unity_flexmdm_ref_pretokenize.sbatch)"; exit 2; }

cd "$REF_ROOT"
echo "=== reference FlexMDM train: steps=$STEPS global_batch=$GLOBAL_BATCH micro=$MICRO gpus=$NPROC seed=$SEED"
echo "pretokenized_root=$PRETOKENIZED_ROOT save_root=$SAVE_ROOT wandb=$WANDB_ENTITY/$WANDB_PROJECT/$RUN_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

srun --ntasks=1 --kill-on-bad-exit=1 python -m torch.distributed.run \
    --standalone --nnodes 1 --nproc_per_node "$NPROC" \
    -m flexmdm.train_fsdp --config-name fsdp_train \
    "trainer.default_local_dir=$SAVE_ROOT" \
    "trainer.wandb.dir=$WANDB_DIR" \
    "trainer.project_name=$WANDB_PROJECT" \
    "trainer.experiment_name=$RUN_NAME" \
    "trainer.seed=$SEED" \
    "trainer.total_training_steps=$STEPS" \
    "trainer.save_checkpoint_steps=$STEPS" \
    "trainer.validation_steps=$STEPS" \
    "trainer.total_epochs=1000" \
    "data.train_batch_size=$GLOBAL_BATCH" \
    "data.micro_batch_size_per_gpu=$MICRO" \
    "data.validation_num_t_strata=8" \
    "model.attn_implementation=sdpa" \
    "fsdp.sharding_strategy=full_shard" \
    "$@"
