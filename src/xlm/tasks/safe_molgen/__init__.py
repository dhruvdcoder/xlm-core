"""Public entry point for SAFE / molecular generation task code.

Implementation lives in :mod:`~xlm.tasks.safe_molgen._safe_molgen_impl` and loads on first use
(attributes other than ``ZINC_LENGTH_REF_FILE`` trigger the import).

Optional CHEMISTRY install::

    pip install "xlm-core[safe]"

For the fuller GenMol / Biomemo / OpenBabel stack (see ``requirements/molgen_requirements.txt``)::

    pip install "xlm-core[molgen]"
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

ZINC_LENGTH_REF_FILE = Path(__file__).resolve().parent / "zinc_len.pkl"

_impl_mod: Any | None = None


def _impl() -> Any:
    global _impl_mod
    if _impl_mod is None:
        _impl_mod = importlib.import_module("xlm.tasks.safe_molgen._safe_molgen_impl")
    return _impl_mod


def __getattr__(name: str) -> Any:
    if name == "ZINC_LENGTH_REF_FILE":
        return ZINC_LENGTH_REF_FILE
    return getattr(_impl(), name)
