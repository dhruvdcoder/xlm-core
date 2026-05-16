"""CLI smoke tests for ``xlm job_type=eval``."""

import subprocess

import pytest


@pytest.mark.cli
@pytest.mark.slow
class TestEvalCLI:
    """Smoke tests for the evaluation CLI."""

    def test_eval_smoke(self, xlm_bin, cli_output_dir):
        """Run a single eval step.

        Requires a checkpoint and experiment config.
        """
        pytest.skip(
            "Requires a tiny experiment config + checkpoint -- "
            "implement when ready"
        )
