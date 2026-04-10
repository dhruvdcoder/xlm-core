#!/usr/bin/env bash
# One-time setup for the ESMFold evaluation venv.
#
# Run on a GPU node — the OpenFold CUDA kernel must be compiled with nvcc
# against the same PyTorch/CUDA pair that will be used at eval time.
#
# Usage:
#   bash evals/protein_eval/setup_eval_venv.sh
#
# Override defaults:
#   PYTHON=python3.10  VENV_DIR=...  DPLM_ROOT=...  TORCH_INDEX_URL=...
#   bash evals/protein_eval/setup_eval_venv.sh
#
# After setup, activate the venv with:
#   source "$VENV_DIR/bin/activate"
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
EVAL_DIR="${REPO_ROOT}/evals/protein_eval"

# Python to build the venv from. Python 3.10 avoids the fair-esm dataclass bug
# on Python 3.11+. If 3.10 is unavailable, the script applies the patch automatically.
PYTHON="${PYTHON:-python3.10}"
if ! command -v "${PYTHON}" &>/dev/null; then
    echo "WARNING: ${PYTHON} not found; falling back to python3. The fair-esm patch will be applied."
    PYTHON=python3
fi

# DPLM submodule root (OpenFold vendored inside it).
DPLM_ROOT="${DPLM_ROOT:-${REPO_ROOT}/vendor/dplm}"
if [ ! -d "${DPLM_ROOT}/vendor/openfold" ]; then
    echo "ERROR: ${DPLM_ROOT}/vendor/openfold not found."
    echo "  If using the submodule: git submodule update --init --recursive"
    echo "  If using a separate clone: DPLM_ROOT=/path/to/dplm bash $0"
    exit 1
fi
OPENFOLD_DIR="${DPLM_ROOT}/vendor/openfold"

# Venv location.
VENV_DIR="${VENV_DIR:-${DPLM_ROOT}/.venv-esmfold}"

# PyTorch wheel index. Adjust cu121 → cu118 / cu124 to match your CUDA toolkit.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

echo "=== ESMFold eval venv setup ==="
echo "  PYTHON      : ${PYTHON}"
echo "  VENV_DIR    : ${VENV_DIR}"
echo "  DPLM_ROOT   : ${DPLM_ROOT}"
echo "  TORCH_INDEX : ${TORCH_INDEX_URL}"
echo

# ── 1. Create venv ────────────────────────────────────────────────────────────
if [ -d "${VENV_DIR}" ]; then
    echo "Venv already exists at ${VENV_DIR} — skipping creation."
else
    "${PYTHON}" -m venv "${VENV_DIR}"
    echo "Created venv: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "Python: $(command -v python)  $(python --version)"

# ── 2. Install PyTorch ────────────────────────────────────────────────────────
echo
echo "=== Installing PyTorch ==="
pip install --quiet --upgrade pip
pip install torch --index-url "${TORCH_INDEX_URL}"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"

# ── 3. Install fair-esm and runtime deps ─────────────────────────────────────
echo
echo "=== Installing fair-esm + deps ==="
pip install fair-esm
pip install -r "${EVAL_DIR}/requirements-esmfold-eval.txt"

# ── 4. Apply Python 3.11+ dataclass patch if needed ──────────────────────────
PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
PY_MAJOR=$(python -c "import sys; print(sys.version_info.major)")
if [ "${PY_MAJOR}" -ge 3 ] && [ "${PY_MINOR}" -ge 11 ]; then
    echo
    echo "=== Applying fair-esm dataclass patch (Python ${PY_MAJOR}.${PY_MINOR}) ==="
    python "${EVAL_DIR}/patch_fair_esm_py311.py"
fi

# ── 5. Build OpenFold CUDA extension ─────────────────────────────────────────
echo
echo "=== Building OpenFold CUDA extension ==="
echo "  Using: $(command -v python)  nvcc: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not on PATH — CPU-only build')"
pip install --quiet "wheel>=0.40" "setuptools>=65"
(cd "${OPENFOLD_DIR}" && python setup.py develop --no-deps)
echo "OpenFold CUDA extension built."

# ── 6. Smoke test ─────────────────────────────────────────────────────────────
echo
echo "=== Smoke test ==="
export PYTHONPATH="${OPENFOLD_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
python - <<'EOF'
import esm
print(f"fair-esm location : {esm.__file__}")
print(f"fair-esm version  : {getattr(esm, '__version__', 'unknown')}")
assert hasattr(esm, "pretrained"), (
    "esm.pretrained not found — wrong 'esm' package installed "
    "(EvolutionaryScale esm, not fair-esm). Use a clean venv."
)
print("esm.pretrained    : OK")

import openfold
print(f"openfold location : {openfold.__file__}")

try:
    import attn_core_inplace_cuda
    print("CUDA extension    : OK")
except ImportError as e:
    print(f"WARNING: attn_core_inplace_cuda not importable: {e}")
    print("  ESMFold will fail at runtime. Re-run on a GPU node with nvcc on PATH.")
EOF

echo
echo "=== Setup complete ==="
echo "Activate the venv with:"
echo "  source ${VENV_DIR}/bin/activate"
