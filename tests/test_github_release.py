"""Tests for .github/release.py (version parse/read/write; no git/gh)."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RELEASE_PY = _REPO_ROOT / ".github" / "release.py"

_spec = importlib.util.spec_from_file_location("github_release", _RELEASE_PY)
assert _spec and _spec.loader
release_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = release_mod
_spec.loader.exec_module(release_mod)


def test_parse_version():
    parts = release_mod.parse_version("0.1.4")
    assert parts.version == "0.1.4"
    assert parts.tag == "v0.1.4"

    parts = release_mod.parse_version("v0.2.0-alpha")
    assert parts.major == "0"
    assert parts.minor == "2"
    assert parts.patch == "0"
    assert parts.suffix == "-alpha"
    assert parts.version == "0.2.0-alpha"


def test_parse_version_rejects_invalid():
    with pytest.raises(ValueError):
        release_mod.parse_version("not-a-version")
    with pytest.raises(ValueError):
        release_mod.parse_version("0.1.4alpha")


def test_write_and_read_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "version.py"
        path.write_text(release_mod.VERSION_PY.read_text(encoding="utf-8"), encoding="utf-8")
        target = release_mod.VersionParts("1", "2", "3", "-rc1")
        release_mod.write_version(target, path=path)
        assert release_mod.read_version(path) == "1.2.3-rc1"


def test_models_version_py_exists():
    assert release_mod.MODELS_VERSION_PY.is_file()


def test_write_and_read_roundtrip_models_version_py():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "version.py"
        path.write_text(
            release_mod.MODELS_VERSION_PY.read_text(encoding="utf-8"), encoding="utf-8"
        )
        target = release_mod.VersionParts("1", "2", "3", "-rc1")
        release_mod.write_version(target, path=path)
        assert release_mod.read_version(path) == "1.2.3-rc1"


def test_read_version_ignores_env_override():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "version.py"
        path.write_text(release_mod.VERSION_PY.read_text(encoding="utf-8"), encoding="utf-8")
        os.environ["XLM_CORE_VERSION_MAJOR"] = "9"
        try:
            assert release_mod.read_version(path) == release_mod.read_version(
                release_mod.VERSION_PY
            )
        finally:
            os.environ.pop("XLM_CORE_VERSION_MAJOR", None)
