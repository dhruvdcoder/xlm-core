"""Unit tests for :mod:`xlm.utils.os`."""

from xlm.utils.os import get_num_processes, is_notebook


def test_get_num_processes_positive():
    n = get_num_processes()
    assert isinstance(n, int)
    assert n > 0


def test_is_notebook_false_outside_ipython():
    # Pytest runs in a normal Python interpreter, not a ZMQ-backed shell.
    assert is_notebook() is False
