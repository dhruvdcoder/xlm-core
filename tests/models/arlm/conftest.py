"""Fixtures specific to ARLM tests."""

import pytest
import torch

from arlm.types_arlm import ARLMBatch


@pytest.fixture()
def arlm_batch(simple_tokenizer, batch_size):
    """A minimal :class:`ARLMBatch` for causal LM training.

    ``target_ids`` is ``input_ids`` shifted left by one; the last position
    and any prompt positions use ``-100`` (ignore index).
    """
    seq_len = 32
    vocab_size = simple_tokenizer.vocab_size

    input_ids = torch.randint(7, vocab_size, (batch_size, seq_len))
    # Shift targets left by 1; pad last position with -100
    target_ids = torch.full_like(input_ids, -100)
    target_ids[:, :-1] = input_ids[:, 1:]

    return ARLMBatch(
        input_ids=input_ids,
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        target_ids=target_ids,
    )
