"""Tests for the confidence-based decoding branch of FlexMDMPredictor2."""

from types import SimpleNamespace

import pytest
import torch

from flexmdm.flexmdm_predictor2 import FlexMDMPredictor2
from flexmdm.noise_flexmdm import FlexMDMNoiseSchedule, LinearSchedule

PAD, EOS, MASK, BOS = 0, 1, 2, 3
VOCAB = 12
FAVORED = 5
PREFIX = [4, 6, 7, 8]
MAX_LEN = 24


class ConstantRateSchedule(LinearSchedule):
    def __init__(self, rate: float):
        self.rate = rate

    def rate_scale_factor(self, t):
        return torch.full_like(t, self.rate)


class FakeModel(torch.nn.Module):
    """Favors one token everywhere and predicts a fixed expected insertion length."""

    def __init__(self, expected_len: float = 1.0):
        super().__init__()
        self.expected_len = expected_len
        self.mask_counts = []

    def forward(self, xt, t, attention_mask):
        self.mask_counts.append((xt == MASK).sum(dim=1).float())
        logits = torch.randn(*xt.shape, VOCAB)
        logits[..., FAVORED] += 5.0
        logits[..., [PAD, MASK]] = -1e4
        length = torch.full(xt.shape, self.expected_len)
        return logits, length


def make_tokenizer():
    return SimpleNamespace(
        pad_token_id=PAD,
        eos_token_id=EOS,
        mask_token_id=MASK,
        bos_token_id=BOS,
        batch_decode=lambda x, skip_special_tokens=True: [""] * len(x),
    )


def make_batch(batch_size: int = 3):
    row = PREFIX + [BOS, EOS]
    input_ids = torch.full((batch_size, MAX_LEN), PAD)
    input_ids[:, : len(row)] = torch.tensor(row)
    fixed = torch.zeros_like(input_ids)
    fixed[:, : len(PREFIX)] = 1
    return {"input_ids": input_ids, "fixed": fixed}


def make_predictor(confidence, unmasking_schedule=None, **kwargs):
    schedule = FlexMDMNoiseSchedule(
        LinearSchedule(), unmasking_schedule or LinearSchedule()
    )
    return FlexMDMPredictor2(
        max_steps=kwargs.pop("max_steps", 6),
        tokenizer=make_tokenizer(),
        model=kwargs.pop("model", FakeModel()),
        noise_schedule=schedule,
        len_predict_type="expectation",
        confidence=confidence,
        **kwargs,
    )


@pytest.mark.parametrize(
    "confidence", ["position", "top_prob", "prob_diff", "entropy", "sampled_prob"]
)
def test_confidence_methods_preserve_fixed_and_special_tokens(confidence):
    torch.manual_seed(0)
    # A huge reveal rate makes the Poisson count exceed the number of masks.
    predictor = make_predictor(
        confidence, unmasking_schedule=ConstantRateSchedule(1e3)
    )
    ids = predictor.predict(make_batch())["ids"]
    assert (ids[:, : len(PREFIX)] == torch.tensor(PREFIX)).all()
    assert ((ids == BOS).sum(dim=1) == 1).all()
    assert ((ids == EOS).sum(dim=1) == 1).all()
    assert not (ids == MASK).any()
    lengths = (ids != PAD).sum(dim=1)
    for row, n in zip(ids, lengths):
        assert not (row[:n] == PAD).any()


def test_sampled_tokens_are_placed():
    torch.manual_seed(0)
    predictor = make_predictor(
        "sampled_prob",
        unmasking_schedule=ConstantRateSchedule(1e3),
        max_steps=3,
    )
    predictor.sampling_function = lambda logits: torch.full(
        logits.shape[:-1], 9, dtype=torch.long
    )
    ids = predictor.predict(make_batch())["ids"]
    generated = ids[:, len(PREFIX) :]
    generated = generated[(generated != PAD) & (generated != BOS) & (generated != EOS)]
    assert (generated == 9).any()
    assert set(generated.tolist()) <= {9, FAVORED}


@pytest.mark.parametrize("scale", [True, False])
def test_reveal_count_scaling(monkeypatch, scale):
    torch.manual_seed(0)
    rate = 2.5
    model = FakeModel(expected_len=2.0)
    predictor = make_predictor(
        "top_prob",
        unmasking_schedule=ConstantRateSchedule(rate),
        model=model,
        scale_unmask_count_by_hazard=scale,
    )
    expected_counts = []
    original_poisson = torch.poisson

    def recording_poisson(x):
        if x.dim() == 1:
            expected_counts.append(x.clone())
            return torch.zeros_like(x)
        return original_poisson(x)

    monkeypatch.setattr(torch, "poisson", recording_poisson)
    predictor.predict(make_batch())

    factor = rate if scale else 1.0
    assert len(expected_counts) == predictor.max_steps - 1
    for step, got in enumerate(expected_counts):
        want = model.mask_counts[step] * predictor.dt * factor
        torch.testing.assert_close(got, want)
    assert any((c > 0).any() for c in expected_counts)


def test_zero_temperature_is_greedy():
    predictor = make_predictor("sampled_prob", temperature=0)
    logits = torch.randn(2, 5, VOCAB)
    assert torch.equal(predictor.sampling_function(logits), logits.argmax(-1))
