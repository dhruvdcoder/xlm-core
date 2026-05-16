"""CLI smoke tests for ``xlm job_type=train``."""

import subprocess

import pytest


@pytest.mark.cli
@pytest.mark.slow
class TestTrainCLI:
    """Smoke tests that invoke the ``xlm`` CLI for training."""

    def test_train_smoke_fast_dev_run(self, xlm_bin, cli_output_dir):
        """Run a single train step via ``trainer.fast_dev_run=true``.

        This test requires a tiny experiment config to be available.
        Adjust the ``experiment=`` override to point to your config.
        """
        pytest.skip(
            "Requires a tiny experiment config -- "
            "implement when configs/experiment/mlm/tiny.yaml exists"
        )
        # Example invocation (uncomment when config is ready):
        # result = subprocess.run(
        #     [
        #         xlm_bin,
        #         "job_type=train",
        #         "experiment=mlm/tiny",
        #         f"paths.output_dir={cli_output_dir}",
        #         "trainer.fast_dev_run=true",
        #     ],
        #     capture_output=True,
        #     text=True,
        #     timeout=120,
        # )
        # assert result.returncode == 0, result.stderr

    def test_train_prints_config_name(self, xlm_bin, cli_output_dir):
        """``job_type=name`` should print the config tree and exit successfully."""
        pytest.skip(
            "Requires a tiny experiment config -- "
            "implement when configs/experiment/mlm/tiny.yaml exists"
        )
