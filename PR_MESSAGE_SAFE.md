# Consolidate SAFE / molgen task code in `xlm.tasks.safe_molgen`

## Motivation

- Single canonical module for SAFE bracket encoding, fragment preprocessing, and post-hoc evaluators (`DeNovoEval`, `FragmentEval`). Deprecated TorchMetrics molgen training metrics were **not** carried forward.
- Align FlexMDM configs and lightning_train SAFE datasets with the learned-noise pipeline (fragment infill, logging fields, checkpoint cadence where copied).
- Remove `src/xlm/tasks/molgen.py` so there is no duplicate or shim layer.

## What changed

### Python

- **Added** [`src/xlm/tasks/safe_molgen.py`](src/xlm/tasks/safe_molgen.py): based on learned-noise `molgen_new.py` (GenMol-aligned `BracketSAFEConverter`, FCD hooks, grouped `FragmentEval`, etc.). Module name avoids clashing with the PyPI **`safe`** package.
- **Deleted** [`src/xlm/tasks/molgen.py`](src/xlm/tasks/molgen.py).
- **Removed** deprecated TorchMetrics path: no `MolGenMetric` / `molgen_update_fn` in `safe_molgen.py`; deleted `src/xlm/configs/lightning_train/metrics/molgen.yaml`. Use **`post_hoc_evaluator`** + `DeNovoEval` / `FragmentEval` for molecule metrics.

### Configs (Hydra)

- **Copied / updated** from learned-noise:
  - `xlm-models/flexmdm/configs/experiment/safe_flexmdm.yaml`, `safe_flexmdm_fragment.yaml`
  - `xlm-models/flexmdm/configs/datamodule/safe_flexmdm.yaml`, `safe_flexmdm_fragment.yaml`
  - `src/xlm/configs/lightning_train/datamodule/safe_fragment.yaml`
  - `src/xlm/configs/lightning_train/datasets/safe_fragment_test.yaml`, `safe_fragment_val.yaml`
  - `datasets/safe.yaml`, `safe_train.yaml`, `safe_val.yaml`, `safe_test.yaml`
  - `tokenizer/safe.yaml`, `post_hoc_evaluator/denovo.yaml`
- **Retargeted** all `xlm.tasks.molgen` / `xlm.tasks.molgen_new` references to **`xlm.tasks.safe_molgen`** (including `qm9_flexmdm` post-hoc evaluator).
- **Fixed** `safe_flexmdm_fragment` datamodule `print_batch_fn`: `lflexmdm...` → **`flexmdm.datamodule_flexmdm.print_batch_flexmdm`**.

### Requirements

- Comment update in [`requirements/molgen_requirements.txt`](requirements/molgen_requirements.txt) (datamol note refers to `xlm.tasks.safe_molgen`). Dependency set already matched learned-noise (`datamol`, `torch_fcd`, `fcd`, etc.); `datasets` remains on core install.

## Config migration cheat sheet

| Old | New |
|-----|-----|
| `xlm.tasks.molgen.*` | `xlm.tasks.safe_molgen.*` |
| `xlm.tasks.molgen_new.*` | `xlm.tasks.safe_molgen.*` |
| `xlm.tasks.safe.*` (internal task module; clashes with PyPI `safe`) | `xlm.tasks.safe_molgen.*` |

Unchanged: YAML keys such as **`molgen_task_name`** (task selector, not a Python module).

## Verification

- `python -m py_compile src/xlm/tasks/safe_molgen.py` (passes).
- Full import test requires a molgen-capable env (`torch`, `rdkit`, `safe-mol`, `datamol`, `pytdc`, …). Example:

  ```bash
  conda activate /scratch3/workspace/dhruveshpate_umass_edu-idlm/xlm-core/.venv_xlm_core  # or your env
  cd xlm-core && PYTHONPATH=src python -c "from xlm.tasks.safe_molgen import DeNovoEval, FragmentEval"
  ```

- Recommended: Hydra compose training config with `experiment=safe_flexmdm` and `experiment=safe_flexmdm_fragment` once `flexmdm` is on the Hydra search path (`xlm-models` discovery).

## Follow-ups

- Delete this file before or after merge if you do not want it in the repo long term.
- Watch for downstream repos that imported `xlm.tasks.molgen` in Python code (xlm-core YAMLs are updated; external callers must switch to `xlm.tasks.safe_molgen`).
