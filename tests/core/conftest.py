"""Fixtures specific to ``tests/core/``."""

import pytest
import torch

from xlm.datamodule import (
    DefaultCollator,
    DefaultCollatorWithPadding,
    DefaultCollatorWithDynamicPadding,
)


@pytest.fixture()
def default_collator(simple_tokenizer, dummy_noise_schedule):
    """A :class:`DefaultCollator` wired to the shared tokenizer."""
    return DefaultCollator(
        tokenizer=simple_tokenizer,
        block_size=32,
        noise_schedule=dummy_noise_schedule,
    )


@pytest.fixture()
def padding_collator(simple_tokenizer, dummy_noise_schedule):
    """A :class:`DefaultCollatorWithPadding` (pads/truncates to block_size)."""
    return DefaultCollatorWithPadding(
        tokenizer=simple_tokenizer,
        block_size=32,
        noise_schedule=dummy_noise_schedule,
    )


@pytest.fixture()
def dynamic_padding_collator(simple_tokenizer, dummy_noise_schedule):
    """A :class:`DefaultCollatorWithDynamicPadding` (pads to longest in batch)."""
    return DefaultCollatorWithDynamicPadding(
        tokenizer=simple_tokenizer,
        block_size=64,
        noise_schedule=dummy_noise_schedule,
    )
