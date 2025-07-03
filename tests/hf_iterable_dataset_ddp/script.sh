#!/bin/bash
#SBATCH --job-name=pl_ddp_test
#SBATCH --nodes=2                     # Adjust number of nodes as needed
#SBATCH --ntasks-per-node=1           # One GPU (process) per node
#SBATCH --cpus-per-task=3             # At least as many dataloader workers as required
#SBATCH --gres=gpu:1                  # Request one GPU per node
#SBATCH --time=00:10:00               # Job runtime (adjust as needed)
#SBATCH --partition=gpu-preempt               # Partition or queue name
#SBATCH -o script.out
#SBATCH --open-mode=append

# Disable Python output buffering.
export PYTHONUNBUFFERED=1
echo "--------------------------------------------------------------------------------------------------------"
echo "SLURM job starting on $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "Current directory: $(pwd)"
ls -l

# Launch the script using srun so that each process starts the Lightning module.
srun script.py
