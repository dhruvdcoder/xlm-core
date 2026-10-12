"""FlexMDM (Dream-Coder) training loss: port of ``genuine-any-order/FlexMDM/flexmdm/trainer.py::FlexMDMProcess``.

Batch contract (reference ``data.py`` layout, see :mod:`flexmdm.datamodule_flexmdm_dream`):

- ``input_ids``  (B, L): ``[BOS] + prompt + [PAD] + answer``, right-padded with PAD.
- ``attention_mask`` (B, L): real tokens (incl. the PAD separator).
- ``prompt_mask`` (B, L): tokens that are never noised (prompt incl. separator).
- ``t`` (B,) optional: diffusion times; sampled here (globally stratified) when absent.

Noising (``flexmdm_process``): every non-prompt token draws hitting times
``t_ins < t_unmask`` from the insertion / unmasking schedules; tokens with
``t < t_ins`` are *deleted* (written as PAD and left-compacted), tokens with
``t_ins <= t < t_unmask`` are *masked*, the rest are kept. Targets: shifted
(Dream-style) cross-entropy on masked positions and a Poisson/Bregman loss
``exp(z) - gaps * z`` on the after-token gap counts, both weighted by the
respective schedule hazard ``alpha'(t) / (1 - alpha(t) + 1e-3)`` and normalized by
``B * max_length``. RNG call order matches the reference (two ``rand_like`` draws
of shape (B, L): insertion then unmasking time).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from xlm.datamodule import Tokenizer
from xlm.utils.rank_zero import RankedLogger

from .noise_flexmdm import FlexMDMNoiseSchedule, Schedule
from .types_flexmdm import FlexMDMModel

logger = RankedLogger(__name__, rank_zero_only=True)


# ---------------------------------------------------------------------------
# Schedule helpers (reference ``schedules.py``)
# ---------------------------------------------------------------------------


def sample_schedule_time_after(start_t: torch.Tensor, schedule: Schedule) -> torch.Tensor:
    """Hitting time after ``start_t``: ``alpha^{-1}(alpha(start) + U * (1 - alpha(start)))``."""
    start_alpha = schedule.at(start_t)
    rand = torch.rand_like(start_t, dtype=torch.float32)
    alpha = start_alpha + rand * (1.0 - start_alpha)
    return schedule.inv(alpha).to(start_t.device)


def sample_schedule_times(
    x1: torch.Tensor, *, insertion_schedule: Schedule, unmasking_schedule: Schedule, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token ``(t_insertion, t_unmasking)`` with ``eps <= t_ins <= t_unmask``."""
    eps_time = x1.new_full(x1.shape, float(eps), dtype=torch.float32)
    insertion_time = sample_schedule_time_after(eps_time, insertion_schedule)
    unmasking_time = sample_schedule_time_after(insertion_time, unmasking_schedule)
    return insertion_time, unmasking_time


def schedule_elbo_weight(schedule: Schedule, t: torch.Tensor, eps: float) -> torch.Tensor:
    """Hazard ``alpha'(t) / (1 - alpha(t) + eps)``; reference caps with ``eps = max(eps, 1e-3)``."""
    return schedule.derivative_at(t) / (1.0 - schedule.at(t) + float(eps))


def sample_globally_stratified_time(
    per_rank_batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-6,
    step: Optional[int] = None,
    base_seed: int = 0,
) -> torch.Tensor:
    """One uniform draw per stratum of ``[0, 1 - eps)`` split into ``B * world_size`` strata.

    With ``step`` given, a permutation seeded by ``base_seed + step`` (identical on all
    ranks) shuffles stratum ownership across ranks.
    """
    global_batch = int(per_rank_batch_size) * int(world_size)
    stratum_width = (1.0 - float(eps)) / float(global_batch)
    start = int(rank) * int(per_rank_batch_size)
    if step is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(base_seed) + int(step))
        permutation = torch.randperm(global_batch, generator=gen)
        stratum_indices = permutation[start : start + int(per_rank_batch_size)]
    else:
        stratum_indices = torch.arange(start, start + int(per_rank_batch_size), dtype=torch.long)
    stratum_indices = stratum_indices.to(device=device, dtype=dtype)
    noise = torch.rand(int(per_rank_batch_size), device=device, dtype=dtype)
    return (stratum_indices + noise) * stratum_width


def poisson_loss(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Poisson NLL up to a constant, in float32: ``exp(z) - x * z``."""
    z32 = z.float()
    return z32.exp() - x.float() * z32


# ---------------------------------------------------------------------------
# Noising process
# ---------------------------------------------------------------------------


def gaps_and_masks(
    x1: torch.Tensor,
    st: torch.Tensor,
    prompt_mask: torch.Tensor,
    x1_attention_mask: torch.Tensor,
    xt_attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """After-token insertion targets (deleted tokens between consecutive survivors) and their mask."""
    x1_attention_mask = x1_attention_mask.bool()
    xt_attention_mask = xt_attention_mask.bool()
    prompt_mask = prompt_mask.bool()
    x1_len = x1_attention_mask.sum(dim=1)
    xt_len = xt_attention_mask.sum(dim=1)

    original_idx = st.clamp(min=0)
    next_original_idx = torch.cat([original_idx[:, 1:], x1_len.unsqueeze(1)], dim=1)
    last_visible = (xt_len - 1).clamp_min(0).unsqueeze(1)
    next_original_idx.scatter_(1, last_visible, x1_len.unsqueeze(1))

    gaps = (next_original_idx - original_idx - 1).clamp(min=0)
    gaps = gaps.masked_fill(~xt_attention_mask, 0)

    prompt_len = prompt_mask.sum(dim=1)
    first_valid_original_idx = (prompt_len - 1).clamp_min(0)
    has_answer = (x1_attention_mask & ~prompt_mask).any(dim=1)
    insertion_mask = (
        xt_attention_mask
        & has_answer.unsqueeze(1)
        & (original_idx >= first_valid_original_idx.unsqueeze(1))
    )
    return gaps, insertion_mask


def flexmdm_process(
    x1: torch.Tensor,
    t: torch.Tensor,
    prompt_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    insertion_schedule: Schedule,
    unmasking_schedule: Schedule,
    mask_id: int,
    pad_id: int,
    eps: float,
) -> Dict[str, torch.Tensor]:
    """Noise ``x1`` to time ``t``; returns ``xt, xt_attention_mask, st, masked_indices, gaps, insertion_mask``."""
    prompt_mask = prompt_mask.bool()
    attention_mask = attention_mask.bool()
    if (prompt_mask & ~attention_mask).any():
        raise ValueError("prompt_mask must be a subset of attention_mask")
    insertion_time, unmasking_time = sample_schedule_times(
        x1, insertion_schedule=insertion_schedule, unmasking_schedule=unmasking_schedule, eps=eps
    )
    deleted = attention_mask & (t[:, None] < insertion_time) & ~prompt_mask
    masked = (
        attention_mask & (t[:, None] >= insertion_time) & (t[:, None] < unmasking_time) & ~prompt_mask
    )
    xt_attention_mask = attention_mask & ~deleted
    xt = torch.where(deleted, pad_id, torch.where(masked, mask_id, x1))
    # survivors first (stable), then the deleted/padding positions
    st = xt_attention_mask.to(torch.int32).argsort(dim=1, descending=True, stable=True)
    xt = torch.gather(xt, 1, st)
    xt_attention_mask = torch.gather(xt_attention_mask, 1, st)
    st[~xt_attention_mask] = -1
    masked_indices = xt.eq(mask_id) & xt_attention_mask
    gaps, insertion_mask = gaps_and_masks(x1, st, prompt_mask, attention_mask, xt_attention_mask)
    return {
        "xt": xt,
        "xt_attention_mask": xt_attention_mask,
        "st": st,
        "masked_indices": masked_indices,
        "gaps": gaps,
        "insertion_mask": insertion_mask,
    }


def flexmdm_losses(
    logits: torch.Tensor,
    log_rate: torch.Tensor,
    *,
    x1: torch.Tensor,
    t: torch.Tensor,
    noised: Dict[str, torch.Tensor],
    insertion_schedule: Schedule,
    unmasking_schedule: Schedule,
    max_length: int,
    weight_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted unmasking CE and insertion Poisson losses, each normalized by ``B * max_length``."""
    scale_factor = x1.shape[0] * int(max_length)
    masked_indices = noised["masked_indices"]
    insertion_mask = noised["insertion_mask"]
    insertion_w = schedule_elbo_weight(insertion_schedule, t, weight_eps)[:, None].expand_as(insertion_mask)
    unmasking_w = schedule_elbo_weight(unmasking_schedule, t, weight_eps)[:, None].expand_as(masked_indices)

    unmask_pred = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)  # position i predicts token i+1
    target_tokens = torch.gather(x1, 1, noised["st"].clamp(min=0))
    unmask_loss = (
        unmasking_w[masked_indices]
        * F.cross_entropy(unmask_pred[masked_indices], target_tokens[masked_indices], reduction="none")
    ).sum() / scale_factor
    insertion_loss = (
        insertion_w[insertion_mask].float()
        * poisson_loss(noised["gaps"][insertion_mask], log_rate[insertion_mask])
    ).sum() / scale_factor
    return unmask_loss, insertion_loss


# ---------------------------------------------------------------------------
# xlm LossFunction
# ---------------------------------------------------------------------------


class FlexMDMDreamLoss:
    """xlm ``LossFunction`` for FlexMDM (Dream-Coder); see module docstring.

    Args:
        noise_schedule: ``FlexMDMNoiseSchedule`` with the training insertion/unmasking schedules.
        max_length: normalizer ``L`` in ``B * L`` (reference ``data.max_length``, 768).
        eps: time floor for hitting times and strata (reference 1e-6).
        weight_eps: hazard denominator floor for the ELBO weights (reference ``max(eps, 1e-3)``).
        stratified_t: sample ``t`` globally stratified across ranks and micro-batches
            (reference trainer). When False, strata span only the local micro-batch
            (reference ``sample_linear_time`` fallback).
        base_seed: seed for the per-step stratum permutation; defaults to the run seed.
    """

    def __init__(
        self,
        noise_schedule: Optional[FlexMDMNoiseSchedule] = None,
        model: Optional[FlexMDMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
        max_length: int = 768,
        eps: float = 1e-6,
        weight_eps: float = 1e-3,
        stratified_t: bool = True,
        base_seed: Optional[int] = None,
    ):
        self.noise_schedule = noise_schedule
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.eps = float(eps)
        self.weight_eps = max(float(weight_eps), float(eps))
        self.stratified_t = bool(stratified_t)
        self.base_seed = base_seed
        self.pl_module = None

    # -- wiring ---------------------------------------------------------------
    def configure(self, pl_module) -> None:
        self.pl_module = pl_module
        if self.base_seed is None:
            self.base_seed = int(getattr(pl_module.config, "seed", 0) or 0)

    @property
    def insertion_schedule(self) -> Schedule:
        assert self.noise_schedule is not None
        return self.noise_schedule.insertion_noise_schedule

    @property
    def unmasking_schedule(self) -> Schedule:
        assert self.noise_schedule is not None
        return self.noise_schedule.unmasking_noise_schedule

    def _dist_info(self) -> Tuple[int, int, Optional[int], int]:
        """(rank, world_size, global_step or None, accumulate_grad_batches)."""
        pl = self.pl_module
        trainer = getattr(pl, "trainer", None) if pl is not None else None
        if trainer is None:
            return 0, 1, None, 1
        try:
            return (
                int(trainer.global_rank),
                int(trainer.world_size),
                int(trainer.global_step),
                int(trainer.accumulate_grad_batches),
            )
        except Exception:  # trainer not attached yet
            return 0, 1, None, 1

    def sample_t(self, batch_size: int, device: torch.device, batch_idx: Optional[int], training: bool) -> torch.Tensor:
        if not self.stratified_t:
            # reference ``sample_linear_time``: strata over the local batch only
            return sample_globally_stratified_time(batch_size, 0, 1, device, eps=self.eps)
        rank, world, step, accum = self._dist_info()
        if not training or step is None:
            # deterministic strata (reference validation uses fixed strata), fresh noise
            return sample_globally_stratified_time(batch_size, rank, world, device, eps=self.eps)
        # Reference: one stratified vector per optimizer step over the per-rank step batch
        # (micro_batch * accumulation), then sliced per micro-batch in order.
        accum = max(1, accum)
        micro_idx = int(batch_idx or 0) % accum
        t_rank = sample_globally_stratified_time(
            batch_size * accum, rank, world, device, eps=self.eps, step=step, base_seed=int(self.base_seed or 0)
        )
        return t_rank[micro_idx * batch_size : (micro_idx + 1) * batch_size]

    # -- loss -----------------------------------------------------------------
    def loss_fn(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None
        x1 = batch["input_ids"].long()
        attention_mask = batch["attention_mask"].bool()
        prompt_mask = batch["prompt_mask"].bool()
        t = batch.get("t")
        training = bool(getattr(self.model, "training", True))
        if t is None:
            t = self.sample_t(x1.shape[0], x1.device, batch_idx, training)
        t = t.to(device=x1.device, dtype=torch.float32)

        noised = flexmdm_process(
            x1, t, prompt_mask, attention_mask,
            insertion_schedule=self.insertion_schedule,
            unmasking_schedule=self.unmasking_schedule,
            mask_id=int(self.tokenizer.mask_token_id),
            pad_id=int(self.tokenizer.pad_token_id),
            eps=self.eps,
        )
        logits, log_rate = self.model(noised["xt"], t, noised["xt_attention_mask"])
        unmask_loss, insertion_loss = flexmdm_losses(
            logits, log_rate, x1=x1, t=t, noised=noised,
            insertion_schedule=self.insertion_schedule,
            unmasking_schedule=self.unmasking_schedule,
            max_length=self.max_length,
            weight_eps=self.weight_eps,
        )
        return {
            "loss": unmask_loss + insertion_loss,
            "unmask_loss": unmask_loss.detach(),
            "insertion_loss": insertion_loss.detach(),
            "t": t.detach(),
            "num_masked": noised["masked_indices"].sum().detach(),
            "num_deleted": (attention_mask.sum() - noised["xt_attention_mask"].sum()).detach(),
        }

    __call__ = loss_fn


__all__ = [
    "FlexMDMDreamLoss",
    "flexmdm_process",
    "flexmdm_losses",
    "gaps_and_masks",
    "sample_schedule_times",
    "sample_globally_stratified_time",
    "schedule_elbo_weight",
    "poisson_loss",
]
