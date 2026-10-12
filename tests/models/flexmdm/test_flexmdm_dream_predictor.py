"""FlexMDMDreamPredictor: collator layout and same-seed trace parity with the reference sampler."""

import pytest
import torch

from flexmdm.configuration_flexmdm_dream import FlexMDMDreamConfig
from flexmdm.datamodule_flexmdm_dream import (
    FlexMDMDreamPredCollator,
    encode_flexmdm_prompt_ids,
)
from flexmdm.model_flexmdm_dream import FlexMDMDreamModel
from flexmdm.noise_flexmdm import FlexMDMNoiseSchedule, PowerSchedule
from flexmdm.predictor_flexmdm_dream import FlexMDMDreamPredictor

from .ref_utils import import_reference_submodule
from .test_flexmdm_dream_model import TINY

PAD, BOS, MASK = TINY["pad_token_id"], TINY["bos_token_id"], TINY["mask_token_id"]
L = 24


class _Tok:
    """Minimal tokenizer stand-in with Dream-like special ids (pad == eos)."""

    pad_token_id = PAD
    bos_token_id = BOS
    eos_token_id = PAD
    mask_token_id = MASK

    def batch_decode(self, ids, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens) for row in ids]

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        if skip_special_tokens:
            ids = [i for i in ids if i not in (PAD, BOS, MASK)]
        return " ".join(str(i) for i in ids)


def _tiny_model(seed: int = 0) -> FlexMDMDreamModel:
    torch.manual_seed(seed)
    config = FlexMDMDreamConfig(**TINY)
    config._attn_implementation = "sdpa"
    model = FlexMDMDreamModel(config, max_length=L).eval()
    # make AdaLN and the insertion head non-trivial
    for layer in model.model.layers:
        layer.rms_norm_modulations.modulation.weight.data.normal_(std=0.05)
    model.insertion_head.proj2.bias.data.fill_(0.5)
    return model


def _batch():
    prompts = [[10, 11, 12, 13], [20, 21]]
    examples = [{"prompt_ids": p, "task_id": f"T/{i}", "sample_k": i} for i, p in enumerate(prompts)]
    return FlexMDMDreamPredCollator(_Tok(), block_size=L)(examples)


def _predictor(model, **kw) -> FlexMDMDreamPredictor:
    schedule = FlexMDMNoiseSchedule(PowerSchedule(1.7), PowerSchedule(2.89))
    defaults = dict(
        max_steps=12,
        tokenizer=_Tok(),
        model=model,
        noise_schedule=schedule,
        inference_insertion_schedule=PowerSchedule(2.9),
        temperature=0.1,
        confidence="top_k",
    )
    defaults.update(kw)
    return FlexMDMDreamPredictor(**defaults)


def test_pred_collator_layout():
    batch = _batch()
    ids, attn, pmask = batch["input_ids"], batch["attention_mask"], batch["prompt_mask"]
    assert ids.shape == (2, L)
    assert ids[0, :6].tolist() == [BOS, 10, 11, 12, 13, PAD]
    assert ids[1, :4].tolist() == [BOS, 20, 21, PAD]
    assert attn[0].sum() == 6 and attn[1].sum() == 4
    assert torch.equal(attn, pmask)
    assert (ids[~attn] == PAD).all()
    assert batch["task_id"] == ["T/0", "T/1"] and batch["sample_k"] == [0, 1]
    with pytest.raises(ValueError):
        encode_flexmdm_prompt_ids(list(range(L)), max_length=L, pad_id=PAD, bos_id=BOS)


def test_predict_outputs_and_prompt_preserved():
    model = _tiny_model()
    pred = _predictor(model, return_history=True)
    batch = _batch()
    torch.manual_seed(0)
    out = pred.predict(batch)
    ids, attn = out["ids"], out["attention_mask"]
    assert ids.shape == (2, L)
    # prompt region untouched, no masks left, pads outside the attended region
    assert torch.equal(ids[batch["prompt_mask"]], batch["input_ids"][batch["prompt_mask"]])
    assert not (ids[attn] == MASK).any()
    assert (ids[~attn] == PAD).all()
    assert len(out["generated_text"]) == 2 and len(out["text"]) == 2
    assert out["generated_length"] == (attn & ~batch["prompt_mask"]).sum(1).tolist()
    assert len(out["history"][0]) == pred.max_steps + 1  # step 0 .. max_steps
    rows = pred.to_dict(batch, out)
    assert set(rows[0]) >= {"text", "generated_text", "ids", "history", "time_taken", "generated_length"}


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),  # published recipe (poisson, T_ins = 1)
        dict(temperature=0.0),
        dict(insertion_temperature=0.6),
        dict(insertion_count_theta=0.5, insertion_count_sampler="floor_bernoulli"),
        dict(confidence="entropy"),
        dict(confidence="origin"),
        dict(inference_insertion_schedule=None),
    ],
)
def test_trace_parity_with_reference_sampler(kwargs):
    ref_inf = import_reference_submodule("inference")
    model = _tiny_model()
    pred = _predictor(model, **kwargs)
    batch = _batch()

    class _RefModel(torch.nn.Module):
        def forward(self, x, t, attention_mask=None):
            logits, log_rate = model(x, t, attention_mask)
            return {"logits": logits, "log_length": log_rate}

    torch.manual_seed(1234)
    ours = pred.sample(
        batch["input_ids"], batch["attention_mask"], batch["prompt_mask"], trace=True
    )
    train_exp = pred.train_insertion_schedule.exp
    infer_exp = pred.insertion_schedule.exp
    torch.manual_seed(1234)
    ref = ref_inf.flexmdm_generate(
        _RefModel(),
        steps=pred.max_steps,
        input_ids=batch["input_ids"],
        mask_id=MASK,
        pad_id=PAD,
        attention_mask=batch["attention_mask"],
        prompt_mask=batch["prompt_mask"],
        temperature=pred.temperature,
        confidence_method=pred.confidence,
        trace=True,
        eps=pred.eps,
        insertion_schedule="power",
        unmasking_schedule="power",
        insertion_exponent=infer_exp,
        unmasking_exponent=pred.unmasking_schedule.exp,
        train_insertion_schedule="power",
        train_insertion_exponent=train_exp,
        insertion_count_sampler=pred.insertion_count_sampler,
        insertion_temperature=pred.insertion_temperature,
        insertion_count_theta=pred.insertion_count_theta,
    )
    assert torch.equal(ours["sequences"], ref["sequences"])
    assert torch.equal(ours["attention_mask"], ref["attention_mask"])
    assert len(ours["history"]) == len(ref["history"])
    for a, b in zip(ours["history"], ref["history"]):
        assert torch.equal(a, b)
    for a, b in zip(ours["insertion_history"], ref["insertion_history"]):
        assert torch.equal(a, b)
    # the run actually inserted something (otherwise parity is vacuous)
    assert (ours["attention_mask"].sum(1) > batch["attention_mask"].sum(1)).any()
