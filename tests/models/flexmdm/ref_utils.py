"""Shared helpers for FlexMDM (Dream-Coder) tests.

``load_reference_flexmdm()`` imports the public ``genuine-any-order/FlexMDM``
package under the alias ``flexmdm_ref`` (so it does not clash with the
xlm-models ``flexmdm`` package). Set ``FLEXMDM_REF_ROOT`` to the directory that
contains the reference ``flexmdm/`` package; tests needing it skip otherwise.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REF_ALIAS = "flexmdm_ref"


def load_reference_flexmdm():
    root = os.environ.get("FLEXMDM_REF_ROOT")
    if not root:
        pytest.skip("FLEXMDM_REF_ROOT not set (path to genuine-any-order/FlexMDM)")
    pkg_dir = Path(root) / "flexmdm"
    init = pkg_dir / "__init__.py"
    if not init.is_file():
        pytest.skip(f"reference flexmdm package not found at {pkg_dir}")
    if REF_ALIAS not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            REF_ALIAS, init, submodule_search_locations=[str(pkg_dir)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[REF_ALIAS] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    return sys.modules[REF_ALIAS]


def import_reference_submodule(name: str):
    load_reference_flexmdm()
    return importlib.import_module(f"{REF_ALIAS}.{name}")
