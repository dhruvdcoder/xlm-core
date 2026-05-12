"""Unit tests for :class:`xlm.utils.checkpoint_with_thinning.ThinningCheckpoint`."""

import pytest

from xlm.utils.checkpoint_with_thinning import ThinningCheckpoint


def _make_callback(**overrides):
    defaults = dict(every_n_train_steps=1000, keep_multiple=10, save_top_k=-1)
    defaults.update(overrides)
    return ThinningCheckpoint(**defaults)


class TestConstructorValidation:
    def test_rejects_custom_filename(self):
        with pytest.raises(Exception, match="custom filename"):
            ThinningCheckpoint(
                every_n_train_steps=1000, filename="my-{step}"
            )

    def test_requires_every_n_train_steps(self):
        with pytest.raises(Exception, match="every_n_train_steps"):
            ThinningCheckpoint()

    def test_rejects_monitor(self):
        with pytest.raises(Exception, match="monitor"):
            ThinningCheckpoint(
                every_n_train_steps=1000, monitor="val/loss"
            )

    def test_rejects_save_top_k_other_than_minus_one(self):
        with pytest.raises(Exception, match="save_top_k"):
            ThinningCheckpoint(
                every_n_train_steps=1000, save_top_k=1
            )


class TestExtractStep:
    # The default Lightning filename (with ``auto_insert_metric_name=False``
    # forced by the constructor) becomes ``"{epoch}-{step}.ckpt"``, e.g.
    # ``"0-1000.ckpt"``.
    def test_parses_standard_filename(self):
        cb = _make_callback()
        assert cb._extract_step("0-1000.ckpt") == 1000

    def test_parses_filename_with_directory(self):
        cb = _make_callback()
        assert cb._extract_step("/tmp/run/2-5000.ckpt") == 5000

    def test_returns_none_for_unparseable(self):
        cb = _make_callback()
        assert cb._extract_step("last.ckpt") is None
        assert cb._extract_step("model.pt") is None


class TestShouldKeepCheckpoint:
    def test_keeps_at_keep_interval(self):
        # every_n_train_steps=1000, keep_multiple=10 -> keep every 10000 steps.
        cb = _make_callback()
        assert cb._should_keep_checkpoint("0-10000.ckpt")
        assert cb._should_keep_checkpoint("0-20000.ckpt")

    def test_does_not_keep_off_interval(self):
        cb = _make_callback()
        assert not cb._should_keep_checkpoint("0-5000.ckpt")
        assert not cb._should_keep_checkpoint("0-11000.ckpt")

    def test_always_keeps_last_global_step(self):
        cb = _make_callback()
        cb._last_global_step_saved = 12345
        assert cb._should_keep_checkpoint("0-12345.ckpt")

    def test_keeps_unparseable_filename(self):
        cb = _make_callback()
        assert cb._should_keep_checkpoint("last.ckpt") is True


class TestStateDictRoundTrip:
    def test_round_trip_keeps_thinning_state(self):
        cb_a = _make_callback(keep_multiple=7)
        sd = cb_a.state_dict()
        assert sd["keep_multiple"] == 7
        assert sd["_step_extraction_pattern"] == r"\d+-(\d+)"

        cb_b = _make_callback(keep_multiple=1)
        cb_b.load_state_dict(sd)
        assert cb_b.keep_multiple == 7
        assert cb_b._step_extraction_pattern == r"\d+-(\d+)"
