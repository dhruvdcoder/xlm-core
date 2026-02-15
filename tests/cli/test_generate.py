"""CLI smoke tests for ``xlm job_type=generate``."""

import subprocess

import pytest


@pytest.mark.cli
@pytest.mark.slow
class TestGenerateCLI:
    """Smoke tests for the generation CLI."""

    def test_generate_smoke(self, xlm_bin, cli_output_dir):
        """Run a short generation pass.

        Requires a checkpoint and experiment config.
        """
        pytest.skip(
            "Requires a tiny experiment config + checkpoint -- "
            "implement when ready"
        )
