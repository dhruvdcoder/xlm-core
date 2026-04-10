#!/usr/bin/env python3
"""Patch fair-esm ESMFold v1 for Python 3.11+ dataclass rules.

Replaces illegal mutable defaults with field(default_factory=...) in:
  - esm/esmfold/v1/trunk.py   (FoldingTrunkConfig.structure_module)
  - esm/esmfold/v1/esmfold.py (ESMFoldConfig.trunk)

Idempotent — safe to run multiple times. Locates files via `import esm`
(does not import esmfold itself, which would trigger the very error being fixed).
"""
from __future__ import annotations

import pathlib
import sys


def _ensure_field_import(text: str) -> str:
    if "from dataclasses import dataclass, field" in text:
        return text
    old = "from dataclasses import dataclass\n"
    if old not in text:
        raise ValueError("expected 'from dataclasses import dataclass' line")
    return text.replace(old, "from dataclasses import dataclass, field\n", 1)


def _patch_trunk(path: pathlib.Path) -> str:
    text = path.read_text()
    if "field(default_factory=StructureModuleConfig)" in text:
        return f"already patched: {path}"
    text = _ensure_field_import(text)
    old = "    structure_module: StructureModuleConfig = StructureModuleConfig()\n"
    new = (
        "    structure_module: StructureModuleConfig = "
        "field(default_factory=StructureModuleConfig)\n"
    )
    if old not in text:
        raise ValueError(f"missing expected line in {path}")
    path.write_text(text.replace(old, new, 1))
    return f"patched: {path}"


def _patch_esmfold(path: pathlib.Path) -> str:
    text = path.read_text()
    if "field(default_factory=FoldingTrunkConfig)" in text:
        return f"already patched: {path}"
    text = _ensure_field_import(text)
    old = "    trunk: T.Any = FoldingTrunkConfig()\n"
    new = "    trunk: T.Any = field(default_factory=FoldingTrunkConfig)\n"
    if old not in text:
        raise ValueError(f"missing expected line in {path}")
    path.write_text(text.replace(old, new, 1))
    return f"patched: {path}"


def main() -> int:
    try:
        import esm
    except ImportError:
        print(
            "ERROR: import esm failed — activate the eval venv first.",
            file=sys.stderr,
        )
        return 1

    root = pathlib.Path(esm.__file__).resolve().parent / "esmfold" / "v1"
    trunk = root / "trunk.py"
    esmfold = root / "esmfold.py"
    for p in (trunk, esmfold):
        if not p.is_file():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    try:
        print(_patch_trunk(trunk))
        print(_patch_esmfold(esmfold))
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
