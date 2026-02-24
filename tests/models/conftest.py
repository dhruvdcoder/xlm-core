"""Shared fixtures for ``tests/models/``."""

import pytest
import torch


@pytest.fixture()
def tiny_mlm_model(tiny_model_kwargs):
    """A tiny :class:`RotaryTransformerMLMModel`."""
    from mlm.model_mlm import RotaryTransformerMLMModel

    return RotaryTransformerMLMModel(**tiny_model_kwargs)


@pytest.fixture()
def tiny_mdlm_model(tiny_model_kwargs):
    """A tiny :class:`MDLMModel`."""
    from mdlm.model_mdlm import MDLMModel

    return MDLMModel(**tiny_model_kwargs)


@pytest.fixture()
def tiny_arlm_model(tiny_model_kwargs):
    """A tiny :class:`RotaryTransformerARLMModel`."""
    from arlm.model_arlm import RotaryTransformerARLMModel

    return RotaryTransformerARLMModel(**tiny_model_kwargs)


@pytest.fixture()
def tiny_ilm_model(tiny_model_kwargs):
    """A tiny :class:`RotaryTransformerILMModel`."""
    from ilm.model_ilm import RotaryTransformerILMModel

    return RotaryTransformerILMModel(**tiny_model_kwargs)


