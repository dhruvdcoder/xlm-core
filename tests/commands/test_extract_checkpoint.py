"""Unit tests for :mod:`xlm.commands.extract_checkpoint`."""

import pytest
from omegaconf import OmegaConf

from xlm.commands.extract_checkpoint import extract_checkpoint


class TestExtractCheckpointValidation:
    """Only the up-front config validation is unit-testable.

    The rest of :func:`extract_checkpoint` instantiates Lightning / Hydra
    machinery and is exercised by integration tests.
    """

    def test_missing_post_training_raises(self):
        cfg = OmegaConf.create({"post_training": None})
        with pytest.raises(ValueError, match="post_training"):
            extract_checkpoint(cfg)

    def test_no_destination_raises(self):
        # Both ``model_state_dict_path`` and ``repo_id`` are unset -> nothing
        # to do with the extracted weights.
        cfg = OmegaConf.create(
            {
                "post_training": {
                    "checkpoint_path": "/tmp/fake.ckpt",
                    "apply_ema": False,
                }
            }
        )
        with pytest.raises(ValueError, match="model_state_dict_path"):
            extract_checkpoint(cfg)
