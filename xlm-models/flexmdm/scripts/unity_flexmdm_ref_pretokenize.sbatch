#!/bin/bash
# Build the reference FlexMDM pre-tokenized code-SFT shards on Unity (CPU job).
#
# Runs the reference pipeline (genuine-any-order/FlexMDM, rsynced to $REF_ROOT) with a per-dataset
# row cap so a 500-2000 step comparison fits on scratch:
#   python -m flexmdm.data tokenize --limit $LIMIT  ->  drop_truncated_rows.py  ->  merge_manifests.py
# The resulting $PRETOKENIZED_ROOT/manifest.json is consumed both by the reference trainer and by
# xlm-core (datamodule=code_sft_flexmdm_dream, FLEXMDM_PRETOKENIZED_ROOT=$PRETOKENIZED_ROOT).
#
#   sbatch xlm-models/flexmdm/scripts/unity_flexmdm_ref_pretokenize.sbatch
#   LIMIT=20000 sbatch ... # smaller
#SBATCH --job-name=flexmdm_pretok
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail

WORKTREE=${WORKTREE:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core-dream-flexmdm}
VENV=${VENV:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core/.venv_xlm_core}
REF_ROOT=${REF_ROOT:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/flexmdm_ref/FlexMDM}
DATA_ROOT=${DATA_ROOT:-/scratch3/workspace/dhruveshpate_umass_edu-idlm/flexmdm_data}
export PRETOKENIZED_ROOT=${PRETOKENIZED_ROOT:-$DATA_ROOT/pretokenized}
export SCRATCH_ROOT=$DATA_ROOT
export BASE_MODEL=${BASE_MODEL:-Dream-org/Dream-Coder-v0-Base-7B}
LIMIT=${LIMIT:-60000}          # rows streamed per source before filtering
DATASETS=${DATASETS:-}         # space-separated subset; empty = all five from the config

module load conda/latest
conda activate "$VENV"
cd "$WORKTREE"
set -a; [ -f .env ] && source .env; [ -f .secrets.env ] && source .secrets.env; set +a
export HF_TOKEN=${HF_TOKEN:-${HF_HUB_KEY:-}}
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
set -u

mkdir -p "$PRETOKENIZED_ROOT"
cd "$REF_ROOT"

# flexmdm.data reads the YAML literally (no OmegaConf), so resolve ${oc.env:...} first.
RESOLVED_CFG=$DATA_ROOT/fsdp_train.resolved.yaml
python - <<'PY'
import os
from omegaconf import OmegaConf
cfg = OmegaConf.load("flexmdm/config/fsdp_train.yaml")
OmegaConf.resolve(cfg)
OmegaConf.save(cfg, os.path.join(os.environ["DATA_ROOT"], "fsdp_train.resolved.yaml"))
print(OmegaConf.to_yaml(cfg.data))
PY

cmd=(python -u -m flexmdm.data tokenize --config "$RESOLVED_CFG" --output-root "$PRETOKENIZED_ROOT"
     --limit "$LIMIT" --log-every 5000 --overwrite)
if [ -n "$DATASETS" ]; then
  # shellcheck disable=SC2206
  ds=($DATASETS); cmd+=(--datasets "${ds[@]}")
fi
echo "=== tokenize: limit=$LIMIT datasets=${DATASETS:-all} -> $PRETOKENIZED_ROOT"
printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}"

echo "=== drop truncated rows"
python scripts/drop_truncated_rows.py --root "$PRETOKENIZED_ROOT"
echo "=== merge manifests"
python scripts/merge_manifests.py --root "$PRETOKENIZED_ROOT" --config "$RESOLVED_CFG"
python - <<PY
import json
m = json.load(open("$PRETOKENIZED_ROOT/manifest.json"))
print("total_written", m.get("total_written"))
for k, v in m["datasets"].items():
    print(f"  {k:36s} kept={v['kept']}")
PY
du -sh "$PRETOKENIZED_ROOT"
