"""Fixtures specific to CLI tests."""

import pytest
import shutil


@pytest.fixture()
def cli_output_dir(tmp_path):
    """A temporary directory for CLI output artifacts."""
    out = tmp_path / "cli_output"
    out.mkdir()
    return out


@pytest.fixture()
def xlm_bin():
    """Path to the ``xlm`` console-script entry point.

    Returns the string ``"xlm"`` (assumes the package is installed in the
    test environment).  Override this fixture if you need a different path.
    """
    path = shutil.which("xlm")
    if path is None:
        pytest.skip("xlm console script not found on PATH")
    return path
