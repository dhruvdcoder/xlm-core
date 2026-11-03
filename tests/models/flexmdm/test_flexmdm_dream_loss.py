"""FlexMDMDreamLoss: same-seed parity with the reference ``FlexMDMProcess`` loss on a tiny model."""

import pytest
import torch

from flexmdm.loss_flexmdm_dream import (
    FlexMDMDreamLoss,
    flexmdm_process,
    sample_globally_stratified_time,
    schedule_elbo_weight,
)
from flexmdm.noise_flexmdm import FlexMDMNoiseSchedule, PowerSchedule

from .ref_utils import import_reference_submodule
from .test_flexmdm_dream_predictor import BOS, L, MASK, PAD, _Tok, _tiny_model

INS_EXP, UNM_EXP = 1.7, 2.89


def _train_batch():
    """Two rows in the reference layout: [BOS] prompt [PAD] answer, right-padded with PAD."""
    rows = [
        ([10, 11, 12], [30, 31, 32, 33, 34, 35, 36, 37]),
        ([20], [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]),
    ]
    ids = torch.full((len(rows), L), PAD, dtype=torch.long)
    attn = torch.zeros((len(rows), L), dtype=torch.bool)
    prompt = torch.zeros((len(rows), L), dtype=torch.bool)
    for i, (p, a) in enumerate(rows):
        seq = [BOS] + p + [PAD] + a
        ids[i, : len(seq)] = torch.tensor(seq)
        attn[i, : len(seq)] = True
        prompt[i, : len(p) + 2] = True
    return {"input_ids": ids, "attention_mask": attn, "prompt_mask": prompt}


def _ref_process(ref_trainer):
    return ref_trainer.FlexMDMProcess(
        vocab_size=64,
        mask_id=MASK,
        pad_id=PAD,
        max_len=L,
        eps=1e-6,
        insertion_schedule="power",
        unmasking_schedule="power",
        insertion_exponent=INS_EXP,
        unmasking_exponent=UNM_EXP,
    )


def _ours():
    schedule = FlexMDMNoiseSchedule(PowerSchedule(INS_EXP), PowerSchedule(UNM_EXP))
    return FlexMDMDreamLoss(noise_schedule=schedule, model=_tiny_model(), tokenizer=_Tok(), max_length=L)


@pytest.mark.parametrize("t_vals", [[0.05, 0.5], [0.3, 0.95], [0.999, 0.001]])
def test_noising_matches_reference(t_vals):
    ref_trainer = import_reference_submodule("trainer")
    process = _ref_process(ref_trainer)
    batch = _train_batch()
    t = torch.tensor(t_vals, dtype=torch.float32)

    torch.manual_seed(7)
    xt, xt_attn, st, x1, masked, gaps, ins_mask = process.flexmdm_process(
        batch["input_ids"], t, batch["prompt_mask"], batch["attention_mask"]
    )
    torch.manual_seed(7)
    ours = flexmdm_process(
        batch["input_ids"], t, batch["prompt_mask"], batch["attention_mask"],
        insertion_schedule=PowerSchedule(INS_EXP), unmasking_schedule=PowerSchedule(UNM_EXP),
        mask_id=MASK, pad_id=PAD, eps=1e-6,
    )
    assert torch.equal(ours["xt"], xt)
    assert torch.equal(ours["xt_attention_mask"], xt_attn.bool())
    assert torch.equal(ours["st"], st)
    assert torch.equal(ours["masked_indices"], masked.bool())
    assert torch.equal(ours["gaps"], gaps)
    assert torch.equal(ours["insertion_mask"], ins_mask.bool())
    # prompt is never noised and survives at the front
    prompt_len = batch["prompt_mask"].sum(1)
    for i in range(2):
        assert torch.equal(ours["xt"][i, : prompt_len[i]], batch["input_ids"][i, : prompt_len[i]])


def test_elbo_weights_match_reference():
    ref_trainer = import_reference_submodule("trainer")
    process = _ref_process(ref_trainer)
    t = torch.tensor([0.0, 0.1, 0.5, 0.9, 0.999])
    ref_ins, ref_unm = process.elbo_weights(t)
    torch.testing.assert_close(schedule_elbo_weight(PowerSchedule(INS_EXP), t, 1e-3), ref_ins)
    torch.testing.assert_close(schedule_elbo_weight(PowerSchedule(UNM_EXP), t, 1e-3), ref_unm)


@pytest.mark.parametrize("seed,t_vals", [(0, [0.1, 0.6]), (1, [0.45, 0.9]), (2, None)])
def test_loss_matches_reference(seed, t_vals):
    ref_trainer = import_reference_submodule("trainer")
    process = _ref_process(ref_trainer)
    loss = _ours()
    model = loss.model
    batch = _train_batch()
    t = None if t_vals is None else torch.tensor(t_vals, dtype=torch.float32)

    def ref_model(xt, tt, attention_mask=None):
        logits, log_rate = model(xt, tt, attention_mask)
        return {"logits": logits, "log_length": log_rate}

    torch.manual_seed(seed)
    ref_unmask, ref_ins = process.loss(ref_model, batch["input_ids"], batch["prompt_mask"], batch["attention_mask"], t=t)
    torch.manual_seed(seed)
    if t is None:
        # reference falls back to ``sample_linear_time`` (local-batch strata) when t is absent
        loss.stratified_t = False
    out = loss(dict(batch, **({} if t is None else {"t": t})))

    torch.testing.assert_close(out["unmask_loss"], ref_unmask.detach())
    torch.testing.assert_close(out["insertion_loss"], ref_ins.detach())
    torch.testing.assert_close(out["loss"], (ref_unmask + ref_ins).detach())
    assert out["loss"].requires_grad
    assert torch.isfinite(out["loss"])


def test_stratified_time_matches_reference():
    ref_sched = import_reference_submodule("schedules")
    for step in (None, 3):
        for rank in (0, 2):
            torch.manual_seed(11)
            ref = ref_sched.sample_globally_stratified_time(6, rank, 4, torch.device("cpu"), step=step, base_seed=5)
            torch.manual_seed(11)
            ours = sample_globally_stratified_time(6, rank, 4, torch.device("cpu"), step=step, base_seed=5)
            torch.testing.assert_close(ours, ref)
            assert ((ours >= 0) & (ours < 1)).all()


def test_micro_batch_slicing_covers_rank_batch():
    """Micro-batches within one optimizer step tile the per-rank stratified vector."""
    loss = _ours()

    class _Trainer:
        global_rank, world_size, global_step, accumulate_grad_batches = 1, 2, 4, 3

    class _PL:
        trainer = _Trainer()
        config = type("C", (), {"seed": 9})()

    loss.configure(_PL())
    torch.manual_seed(0)
    full = sample_globally_stratified_time(2 * 3, 1, 2, torch.device("cpu"), step=4, base_seed=9)
    torch.manual_seed(0)
    parts = []
    for micro_idx in range(3):
        torch.manual_seed(0)
        parts.append(loss.sample_t(2, torch.device("cpu"), batch_idx=micro_idx, training=True))
    torch.testing.assert_close(torch.cat(parts), full)
