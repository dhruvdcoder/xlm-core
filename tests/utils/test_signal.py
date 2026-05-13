"""Unit tests for :mod:`xlm.utils.signal`."""

import logging
import signal as _signal

import pytest

from xlm.utils.signal import print_signal_handlers, remove_handlers


@pytest.fixture()
def caplog_warnings(caplog):
    caplog.set_level(logging.WARNING, logger="xlm.utils.signal")
    return caplog


class TestPrintSignalHandlers:
    def test_logs_one_line_per_signal(self, caplog_warnings):
        print_signal_handlers(prefix="probe")
        # The implementation prints one line per signal in [SIGTERM, SIGINT,
        # SIGCONT, SIGUSR1, SIGUSR2].
        assert (
            sum("Signal" in r.message for r in caplog_warnings.records) == 5
        )

    def test_prefix_is_included(self, caplog_warnings):
        print_signal_handlers(prefix="probe")
        assert any("probe" in r.message for r in caplog_warnings.records)


class TestRemoveHandlers:
    def test_restores_default_handler(self, caplog_warnings):
        # Capture the existing handler so we can restore it after the test.
        original = _signal.getsignal(_signal.SIGUSR1)
        try:
            # Install a non-default sentinel handler first.
            _signal.signal(_signal.SIGUSR1, lambda *a: None)
            assert _signal.getsignal(_signal.SIGUSR1) is not _signal.SIG_DFL
            remove_handlers([_signal.SIGUSR1], prefix="t")
            assert _signal.getsignal(_signal.SIGUSR1) == _signal.SIG_DFL
        finally:
            _signal.signal(_signal.SIGUSR1, original)

    def test_logs_for_each_signal(self, caplog_warnings):
        original = _signal.getsignal(_signal.SIGUSR2)
        try:
            remove_handlers([_signal.SIGUSR2], prefix="probe")
            assert any(
                "probe" in r.message and "Removing" in r.message
                for r in caplog_warnings.records
            )
        finally:
            _signal.signal(_signal.SIGUSR2, original)
