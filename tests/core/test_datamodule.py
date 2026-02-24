"""Unit tests for core data-module components in ``xlm.datamodule``."""

import pytest

from xlm.datamodule import (
    TextDataModule,
    DatasetManager,
)


class TestDatasetManager:
    """Smoke tests for :class:`DatasetManager`.

    These tests verify the public interface without downloading real data.
    Override or parametrise ``dataset_name`` to test with actual HF datasets.
    """

    @pytest.mark.slow
    def test_prepare_and_setup(self, tmp_path):
        """Placeholder: requires a real or mocked HF dataset."""
        pytest.skip(
            "Requires a small HF dataset or mock -- implement when ready"
        )


class TestTextDataModule:
    """Smoke tests for :class:`TextDataModule`."""

    @pytest.mark.slow
    def test_train_dataloader_returns_batches(self):
        """Placeholder: requires a fully wired config + dataset."""
        pytest.skip("Requires full Hydra config -- implement when ready")
