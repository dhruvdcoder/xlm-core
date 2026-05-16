"""Fixtures specific to ILM tests."""

import pytest
import torch


@pytest.fixture()
def ilm_batch(simple_tokenizer, batch_size):
    """A minimal ILM-style batch.

    The ILM loss expects a sparse ``target_ids`` tensor of shape
    ``(batch, seq_len, vocab_size)`` (counts of dropped tokens at each
    position) plus ``n_drops`` and ``cls_position``.  Building a realistic
    one requires the full collation pipeline, so here we create a simplified
    version for model-level tests.
    """
    seq_len = 32
    vocab_size = simple_tokenizer.vocab_size
    input_ids = torch.randint(7, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
