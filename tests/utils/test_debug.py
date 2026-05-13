"""Unit tests for :func:`xlm.utils.debug.set_flags`."""

from omegaconf import OmegaConf

import xlm.flags as flags
from xlm.utils.debug import set_flags


class TestSetFlags:
    def test_no_global_flags_is_noop(self):
        cfg = OmegaConf.create({"unrelated": True})
        # Should not raise and should not mutate xlm.flags.
        set_flags(cfg)

    def test_sets_existing_flag(self):
        # ``DEBUG`` exists in xlm.flags and defaults to False.
        original = flags.DEBUG
        try:
            cfg = OmegaConf.create({"global_flags": {"DEBUG": True}})
            set_flags(cfg)
            assert flags.DEBUG is True
        finally:
            flags.DEBUG = original

    def test_sets_multiple_flags(self):
        originals = {
            "DEBUG": flags.DEBUG,
            "DEBUG_OVERFIT": flags.DEBUG_OVERFIT,
        }
        try:
            cfg = OmegaConf.create(
                {
                    "global_flags": {
                        "DEBUG": True,
                        "DEBUG_OVERFIT": True,
                    }
                }
            )
            set_flags(cfg)
            assert flags.DEBUG is True
            assert flags.DEBUG_OVERFIT is True
        finally:
            for name, val in originals.items():
                setattr(flags, name, val)

    def test_can_introduce_new_flag(self):
        # ``set_flags`` writes straight into ``xlm.flags.__dict__``.
        try:
            cfg = OmegaConf.create({"global_flags": {"NEW_TEST_FLAG": 42}})
            set_flags(cfg)
            assert flags.__dict__["NEW_TEST_FLAG"] == 42
        finally:
            flags.__dict__.pop("NEW_TEST_FLAG", None)
