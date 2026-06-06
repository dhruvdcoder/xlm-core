"""Tests for FlexMDM tokenizer guards and Qwen pad registration."""

from types import SimpleNamespace

import pytest

from flexmdm.loss_flexmdm import FlexMDMLoss
from flexmdm.noise_flexmdm import FlexMDMNoiseSchedule, LinearSchedule
from flexmdm.predictor_flexmdm import FlexMDMPredictor
from flexmdm.tokenizer_guards import require_distinct_pad_and_eos
from xlm.datamodule import load_auto_tokenizer


@pytest.fixture()
def qwen_tokenizer():
    return load_auto_tokenizer(
        "Qwen/Qwen2-0.5B",
        special_tokens={"mask_token": "<|mask|>", "pad_token": "<|pad|>"},
    )


def test_load_auto_tokenizer_qwen_pad_distinct_from_eos(qwen_tokenizer):
    assert qwen_tokenizer.pad_token == "<|pad|>"
    assert qwen_tokenizer.pad_token_id != qwen_tokenizer.eos_token_id


def test_require_distinct_pad_and_eos_raises_when_equal():
    tok = SimpleNamespace(pad_token_id=5, eos_token_id=5)
    with pytest.raises(ValueError, match="pad_token_id != eos_token_id"):
        require_distinct_pad_and_eos(tok)


def test_require_distinct_pad_and_eos_passes_when_distinct():
    tok = SimpleNamespace(pad_token_id=0, eos_token_id=10)
    require_distinct_pad_and_eos(tok)


def test_flexmdm_loss_raises_when_pad_equals_eos():
    tok = SimpleNamespace(pad_token_id=1, eos_token_id=1, mask_token_id=2)
    schedule = FlexMDMNoiseSchedule(LinearSchedule(), LinearSchedule())
    with pytest.raises(ValueError, match="pad_token_id != eos_token_id"):
        FlexMDMLoss(noise_schedule=schedule, tokenizer=tok)


def test_flexmdm_predictor_raises_when_pad_equals_eos():
    tok = SimpleNamespace(pad_token_id=1, eos_token_id=1, mask_token_id=2)
    schedule = FlexMDMNoiseSchedule(LinearSchedule(), LinearSchedule())
    with pytest.raises(ValueError, match="pad_token_id != eos_token_id"):
        FlexMDMPredictor(
            max_steps=4,
            tokenizer=tok,
            noise_schedule=schedule,
        )
