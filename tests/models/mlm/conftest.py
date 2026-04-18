"""Fixtures specific to MLM tests."""

import pytest
import torch

from mlm.types_mlm import MLMBatch


@pytest.fixture()
def mlm_batch(simple_tokenizer, batch_size):
    """A minimal :class:`MLMBatch` with random masked inputs."""
    seq_len = 32
    vocab_size = simple_tokenizer.vocab_size
    mask_id = simple_tokenizer.mask_token_id

    target_ids = torch.randint(7, vocab_size, (batch_size, seq_len))
    # Mask roughly 15% of positions
    mask = torch.rand(batch_size, seq_len) < 0.15
    input_ids = target_ids.clone()
    input_ids[mask] = mask_id

    return MLMBatch(
        input_ids=input_ids,
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        target_ids=target_ids,
    )
