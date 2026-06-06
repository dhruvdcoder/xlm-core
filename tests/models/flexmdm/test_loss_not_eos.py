"""Tests that get_noised_sequence never deletes or masks EOS."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from flexmdm.loss_flexmdm import get_noised_sequence
from flexmdm.noise_flexmdm import FlexMDMNoiseSchedule, LinearSchedule


def test_get_noised_sequence_preserves_eos_when_fixed_zero_at_eos():
    pad, mask, eos = 0, 99, 10
    # [content, content, eos, pad, pad] — EOS not in fixed (simulates mis-fixed collate)
    x1 = torch.tensor([[1, 2, eos, pad, pad]])
    fixed = torch.tensor([[0, 0, 0, 0, 0]])

    schedule = FlexMDMNoiseSchedule(LinearSchedule(), LinearSchedule())
    # Force all positions (including EOS) to be deletion-eligible without not_eos
    schedule.insertion_noise_schedule.sample = MagicMock(
        return_value=torch.full((1, 5), 0.99)
    )
    schedule.unmasking_noise_schedule.sample_truncated = MagicMock(
        return_value=torch.full((1, 5), 0.5)
    )

    t = torch.tensor([0.5])
    tokenizer = SimpleNamespace(
        pad_token_id=pad, mask_token_id=mask, eos_token_id=eos
    )

    _, xt, _, _, _ = get_noised_sequence(
        x1, 1, 5, t, fixed, schedule, tokenizer
    )

    # EOS must survive (other tokens may be deleted and squeezed left)
    assert (xt[0] == eos).any()
