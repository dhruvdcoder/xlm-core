"""Fixtures specific to FRM tests."""

import pytest
import torch

from frm.types_frm import FRMBatch


@pytest.fixture()
def frm_batch(simple_tokenizer, batch_size):
    """A minimal ``FRMBatch`` with clamp/clue fields."""
    seq_len = 16
    vocab_size = simple_tokenizer.vocab_size
    mask_id = simple_tokenizer.mask_token_id
    pad_id = simple_tokenizer.pad_token_id

    target_ids = torch.randint(7, vocab_size, (batch_size, seq_len))
    clamp_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    clamp_mask[:, :4] = True
    clue_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    clue_ids[:, :4] = target_ids[:, :4]
    input_ids = target_ids.clone()
    input_ids[:, 4:6] = mask_id

    return FRMBatch(
        input_ids=input_ids,
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        target_ids=target_ids,
        clamp_mask=clamp_mask,
        clue_ids=clue_ids,
    )
