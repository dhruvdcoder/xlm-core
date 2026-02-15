"""CLI tests for the ``xlm-scaffold`` command."""

import subprocess

import pytest


@pytest.mark.cli
class TestScaffoldCLI:
    """Tests for the ``xlm-scaffold`` model scaffolding command."""

    def test_scaffold_help_exits_zero(self):
        """``xlm-scaffold --help`` should exit 0."""
        result = subprocess.run(
            ["xlm-scaffold", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()

    @pytest.mark.slow
    def test_scaffold_creates_model_directory(self, tmp_path):
        """``xlm-scaffold`` should create the expected directory structure."""
        pytest.skip(
            "Implement when scaffold_model expectations are finalized"
        )
