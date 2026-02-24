"""Fixtures specific to MDLM tests."""

import pytest
import torch

from mdlm.types_mdlm import MDLMBatch


@pytest.fixture()
def mdlm_batch(simple_tokenizer, batch_size):
    """A minimal ``MDLMBatch`` with noise fields."""
    seq_len = 32
    vocab_size = simple_tokenizer.vocab_size
    mask_id = simple_tokenizer.mask_token_id

    target_ids = torch.randint(7, vocab_size, (batch_size, seq_len))
    mask = torch.rand(batch_size, seq_len) < 0.15
    input_ids = target_ids.clone()
    input_ids[mask] = mask_id

    t = torch.rand(batch_size)

    return MDLMBatch(
        input_ids=input_ids,
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        target_ids=target_ids,
        noise_rate=t,           # simplified; real noise_rate comes from schedule
        total_noise=t,          # simplified
        t=t,
    )
