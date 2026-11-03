"""FlexMDM (Dream-Coder) sampler: port of ``genuine-any-order/FlexMDM/flexmdm/inference.py``.

``FlexMDMDreamPredictor.predict`` is a step-for-step port of ``flexmdm_generate``
with the published decoding path (confidence-based top-k / entropy unmasking or
``origin`` Bernoulli unmasking, Poisson insertion after tokens, count-preserving
insertion temperature, inference-time schedule reparameterization). The
"force-unmask" / "meaningful-replace" tricks (all off in the released evaluation)
are not ported. Random-number consumption order matches the reference, so with
the same seed, model outputs and inputs the generated traces are identical.

Batch contract (see :class:`flexmdm.datamodule_flexmdm_dream.FlexMDMDreamPredCollator`):

- ``input_ids``: ``[BOS] + prompt + [PAD]`` separator, right-padded with PAD to ``L``.
- ``attention_mask`` / ``prompt_mask``: ``True`` over the prefix (incl. separator).
"""

from __future__ import annotations

import time
from itertools import cycle
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Integer
from torch import Tensor as TT

from xlm.datamodule import Tokenizer
from xlm.harness import Predictor, PredictorHistoryMixin
from xlm.utils.rank_zero import RankedLogger

from .noise_flexmdm import FlexMDMNoiseSchedule, Schedule, reparameterize_time
from .types_flexmdm import FlexMDMModel, FlexMDMPredictionDict

logger = RankedLogger(__name__, rank_zero_only=True)

POISSON_INSERTION_COUNT_SAMPLER = "poisson"
FLOOR_BERNOULLI_INSERTION_COUNT_SAMPLER = "floor_bernoulli"
ConfidenceMethod = Literal["top_k", "top_k_probability", "entropy", "origin"]


# ---------------------------------------------------------------------------
# Schedule helpers (reference ``schedules.schedule_hazard_rate``)
# ---------------------------------------------------------------------------


def schedule_hazard_rate(schedule: Schedule, t: torch.Tensor, eps: float) -> torch.Tensor:
    """``alpha'(t) / (1 - alpha(t) + eps)`` (reference formula, incl. the eps floor)."""
    return schedule.derivative_at(t) / (1.0 - schedule.at(t) + float(eps))


# ---------------------------------------------------------------------------
# Sampling primitives (reference ``inference.py``)
# ---------------------------------------------------------------------------


def _sample_floor_plus_bernoulli(expected_count: torch.Tensor) -> torch.Tensor:
    expected_count = expected_count.float()
    base_count = torch.floor(expected_count)
    fractional_count = expected_count - base_count
    bernoulli_count = torch.bernoulli(fractional_count)
    return (base_count + bernoulli_count).long()


def _normalize_insertion_count_sampler(sampler: Optional[str]) -> str:
    if sampler is None:
        return POISSON_INSERTION_COUNT_SAMPLER
    normalized = str(sampler).strip().lower().replace("-", "_")
    if normalized == POISSON_INSERTION_COUNT_SAMPLER:
        return POISSON_INSERTION_COUNT_SAMPLER
    if normalized in {
        FLOOR_BERNOULLI_INSERTION_COUNT_SAMPLER,
        "bernoulli",
        "floor_plus_bernoulli",
    }:
        return FLOOR_BERNOULLI_INSERTION_COUNT_SAMPLER
    raise ValueError(
        "insertion_count_sampler must be one of {'poisson', 'floor_bernoulli'}, "
        "got " + repr(sampler) + "."
    )


def _sample_insertion_counts(
    expected_count: torch.Tensor, *, insertion_count_sampler: str
) -> torch.Tensor:
    sampler = _normalize_insertion_count_sampler(insertion_count_sampler)
    if sampler == POISSON_INSERTION_COUNT_SAMPLER:
        return torch.poisson(expected_count).long()
    return _sample_floor_plus_bernoulli(expected_count)


def gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample token ids with Gumbel-max; ``temperature <= 0`` gives argmax."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    logits = logits.float() / float(temperature)
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise.clamp_min(1e-8)) + 1e-8)
    return (logits + gumbel).argmax(dim=-1)


def _sample_x0_and_confidence(
    logits: torch.Tensor, *, temperature: float, confidence_method: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    sampled_tokens = gumbel_sample(logits, temperature=temperature)
    probs = F.softmax(logits.float(), dim=-1)
    if confidence_method in {"top_k", "top_k_probability"}:
        confidence = probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    elif confidence_method == "entropy":
        log_probs = probs.clamp_min(1e-30).log()
        confidence = (probs * log_probs).sum(dim=-1)  # == -H
    else:
        raise ValueError(
            "confidence_method must be one of {'top_k', 'top_k_probability', "
            "'entropy', 'origin'}, got " + repr(confidence_method) + "."
        )
    return sampled_tokens, confidence


def adaptive_unmask_step(
    xt: torch.Tensor,
    logits: torch.Tensor,
    mask_indices: torch.Tensor,
    *,
    rate: torch.Tensor,
    dt: float,
    temperature: float,
    confidence_method: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Confidence-based any-order unmasking; returns ``(new_xt, confidence)``."""
    sampled_tokens, confidence = _sample_x0_and_confidence(
        logits, temperature=temperature, confidence_method=confidence_method
    )
    min_score = torch.finfo(confidence.dtype).min
    confidence = confidence.masked_fill(~mask_indices, min_score)
    masked_count = mask_indices.sum(dim=1).to(dtype=rate.dtype)
    unmask_count = torch.poisson(masked_count * rate * float(dt)).long()

    new_xt = xt.clone()
    for batch_idx in range(xt.shape[0]):
        candidates = int(mask_indices[batch_idx].sum().item())
        k = min(int(unmask_count[batch_idx].item()), candidates)
        if k > 0:
            select_idx = confidence[batch_idx].topk(k=k).indices
            new_xt[batch_idx, select_idx] = sampled_tokens[batch_idx, select_idx]
    return new_xt, confidence


def origin_unmask_step(
    xt: torch.Tensor,
    logits: torch.Tensor,
    mask_indices: torch.Tensor,
    *,
    rate: torch.Tensor,
    dt: float,
    temperature: float,
) -> torch.Tensor:
    """Independent per-position Bernoulli unmask (Dream-Coder ``origin`` alg)."""
    sampled_tokens = gumbel_sample(logits, temperature=temperature)
    p_transfer = 1.0 - torch.exp(-rate * float(dt))
    p_transfer = p_transfer.clamp(min=0.0, max=1.0).to(xt.device)
    p_broadcast = p_transfer.view(-1, 1).expand_as(mask_indices).float()
    fire = torch.rand_like(p_broadcast) < p_broadcast
    replace = mask_indices & fire
    new_xt = xt.clone()
    new_xt[replace] = sampled_tokens[replace]
    return new_xt


def valid_insertion_gaps(
    *, prompt_len: torch.Tensor, xt_len: torch.Tensor, max_len: int, device: torch.device
) -> torch.Tensor:
    """Valid after-token insertion gaps: real tokens from the last prompt token on."""
    pos = torch.arange(max_len, device=device).view(1, max_len)
    first_allowed = (prompt_len - 1).clamp_min(0).view(-1, 1)
    real_token = pos < xt_len.view(-1, 1)
    after_prompt_boundary = pos >= first_allowed
    has_prompt_anchor = prompt_len.view(-1, 1) > 0
    return real_token & after_prompt_boundary & has_prompt_anchor


def cap_insertions_to_capacity(
    insertions: torch.Tensor, capacity: torch.Tensor
) -> torch.Tensor:
    """Greedily cap per-gap insertions so the sequence never exceeds ``L``."""
    capped = torch.zeros_like(insertions)
    for batch_idx in range(insertions.shape[0]):
        remaining = int(capacity[batch_idx].item())
        if remaining <= 0:
            continue
        for pos_idx in range(insertions.shape[1]):
            count = min(int(insertions[batch_idx, pos_idx].item()), remaining)
            if count > 0:
                capped[batch_idx, pos_idx] = count
                remaining -= count
            if remaining <= 0:
                break
    return capped


def apply_after_token_insertions(
    xt: torch.Tensor,
    attention_mask: torch.Tensor,
    insertions: torch.Tensor,
    *,
    pad_id: int,
    mask_id: int,
    carry: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Insert ``insertions[b, i]`` mask tokens after token ``i``; returns ``(xt, attn, is_inserted, carry)``."""
    batch_size, max_len = xt.shape
    device = xt.device
    pos = torch.arange(max_len, device=device).view(1, max_len).expand_as(xt)
    batch = torch.arange(batch_size, device=device).view(batch_size, 1).expand_as(xt)

    attention_mask = attention_mask.bool()
    xt_len = attention_mask.sum(dim=1)
    total_insertions = insertions.sum(dim=1)
    new_len = xt_len + total_insertions

    new_xt = torch.full_like(xt, pad_id)
    new_attention_mask = torch.zeros_like(attention_mask)
    nonpad_region = pos < new_len.view(batch_size, 1)
    new_xt[nonpad_region] = mask_id
    new_attention_mask[nonpad_region] = True
    is_inserted = nonpad_region.clone()

    shift_before_token = insertions.cumsum(dim=1) - insertions
    new_pos = pos + shift_before_token
    original_token = pos < xt_len.view(batch_size, 1)
    new_xt[batch[original_token], new_pos[original_token]] = xt[original_token]
    is_inserted[batch[original_token], new_pos[original_token]] = False

    new_carry: Optional[torch.Tensor] = None
    if carry is not None:
        new_carry = torch.zeros_like(carry)
        new_carry[batch[original_token], new_pos[original_token]] = carry[original_token]
    return new_xt, new_attention_mask, is_inserted, new_carry


def expected_insertion_counts(
    log_length_pred: torch.Tensor,
    hazard_rate: torch.Tensor,
    dt: float,
    valid_gaps: torch.Tensor,
    *,
    insertion_temperature: float,
) -> torch.Tensor:
    """Per-gap expected counts with count-preserving placement temperature (App. D.5)."""
    g = log_length_pred.clamp(max=15.0).float()
    neg_inf = torch.finfo(g.dtype).min
    g_valid = g.masked_fill(~valid_gaps, neg_inf)
    lambda_total = (
        g_valid.exp().sum(dim=1, keepdim=True) * hazard_rate[:, None] * float(dt)
    )
    if float(insertion_temperature) == 1.0:
        placement = torch.softmax(g_valid, dim=1)
    else:
        placement = torch.softmax(g_valid / float(insertion_temperature), dim=1)
    expected = lambda_total * placement
    return torch.where(lambda_total > 0, expected, torch.zeros_like(expected))


def insertion_step(
    xt: torch.Tensor,
    attention_mask: torch.Tensor,
    log_length_pred: torch.Tensor,
    *,
    prompt_len: torch.Tensor,
    rate: torch.Tensor,
    dt: float,
    pad_id: int,
    mask_id: int,
    insertion_count_sampler: str = POISSON_INSERTION_COUNT_SAMPLER,
    insertion_temperature: float = 1.0,
    insertion_count_theta: float = 1.0,
    insertion_carry: Optional[torch.Tensor] = None,
    flush_carry: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One insertion (tau-leaping) step; returns ``(xt, attn, is_inserted, carry)``."""
    batch_size, max_len = xt.shape
    attention_mask = attention_mask.bool()
    xt_len = attention_mask.sum(dim=1)
    if insertion_carry is None:
        insertion_carry = torch.zeros_like(xt, dtype=torch.float32)

    valid_gaps = valid_insertion_gaps(
        prompt_len=prompt_len, xt_len=xt_len, max_len=max_len, device=xt.device
    )
    theta = float(insertion_count_theta)
    temperature = float(insertion_temperature)
    new_carry = insertion_carry

    if theta >= 1.0 and temperature == 1.0:
        # Bit-exact published recipe: per-gap independent Poisson at rate exp(g) * hazard.
        insertion_rate = log_length_pred.clamp(max=15.0).float().exp() * rate[:, None]
        insertions = _sample_insertion_counts(
            insertion_rate * float(dt), insertion_count_sampler=insertion_count_sampler
        )
        insertions = insertions.masked_fill(~valid_gaps, 0)
    else:
        expected = expected_insertion_counts(
            log_length_pred, rate, dt, valid_gaps, insertion_temperature=temperature
        )
        stochastic = _sample_insertion_counts(
            theta * expected, insertion_count_sampler=insertion_count_sampler
        )
        new_carry = insertion_carry + (1.0 - theta) * expected
        if flush_carry:
            deterministic = torch.round(new_carry).long()
        else:
            deterministic = torch.floor(new_carry).long()
        new_carry = new_carry - deterministic.float()
        insertions = stochastic + deterministic
        insertions = insertions.masked_fill(~valid_gaps, 0)

    insertions = cap_insertions_to_capacity(insertions, capacity=max_len - xt_len)

    if insertions.sum().item() == 0:
        is_inserted = torch.zeros_like(attention_mask, dtype=torch.bool)
        return xt, attention_mask, is_inserted, new_carry
    return apply_after_token_insertions(
        xt, attention_mask, insertions, pad_id=pad_id, mask_id=mask_id, carry=new_carry
    )


def infer_prompt_attention_mask(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Prompt + internal PAD separator mask when only ``input_ids`` are given."""
    batch_size, max_len = input_ids.shape
    pos = torch.arange(max_len, device=input_ids.device).view(1, max_len)
    prompt_token_len = input_ids.ne(pad_id).sum(dim=1)
    initial_len = torch.minimum(
        prompt_token_len + prompt_token_len.lt(max_len).long(),
        prompt_token_len.new_full(prompt_token_len.shape, max_len),
    )
    return pos < initial_len.view(batch_size, 1)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class FlexMDMDreamPredictor(
    torch.nn.Module,
    PredictorHistoryMixin,
    Predictor[Dict[str, Any], FlexMDMPredictionDict],
):
    """FlexMDM (Dream-Coder) sampler implementing the xlm ``Predictor`` protocol.

    Published HumanEval setting: ``max_steps=512``, ``temperature=0.1``,
    ``confidence="top_k"``, train schedules power(1.7)/power(2.89) via
    ``noise_schedule``, ``inference_insertion_schedule=PowerSchedule(2.9)``,
    ``insertion_temperature=1.0`` (0.6 for MBPP), ``insertion_count_sampler="poisson"``.
    """

    def __init__(
        self,
        max_steps: int,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[FlexMDMModel] = None,
        noise_schedule: Optional[FlexMDMNoiseSchedule] = None,
        inference_insertion_schedule: Optional[Schedule] = None,
        temperature: float = 0.1,
        confidence: ConfidenceMethod = "top_k",
        insertion_temperature: float = 1.0,
        insertion_count_sampler: str = POISSON_INSERTION_COUNT_SAMPLER,
        insertion_count_theta: float = 1.0,
        use_midpoint_hazard_rate: bool = False,
        eps: float = 1e-6,
        return_history: bool = False,
        skip_special_tokens_in_history: bool = True,
        progress_bar: bool = False,
    ):
        super().__init__()
        self.init_history(return_history=return_history, decode_fn=self.decode)
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")
        if not float(insertion_temperature) > 0.0:
            raise ValueError(
                f"insertion_temperature must be > 0, got {insertion_temperature}."
            )
        if not 0.0 <= float(insertion_count_theta) <= 1.0:
            raise ValueError(
                f"insertion_count_theta must be in [0, 1], got {insertion_count_theta}."
            )
        _normalize_insertion_count_sampler(insertion_count_sampler)
        self.max_steps = int(max_steps)
        self.tokenizer = tokenizer
        self.model = model
        self.noise_schedule = noise_schedule
        self.inference_insertion_schedule = inference_insertion_schedule
        self.temperature = float(temperature)
        self.confidence = confidence
        self.insertion_temperature = float(insertion_temperature)
        self.insertion_count_sampler = insertion_count_sampler
        self.insertion_count_theta = float(insertion_count_theta)
        self.use_midpoint_hazard_rate = bool(use_midpoint_hazard_rate)
        self.eps = float(eps)
        self.skip_special_tokens_in_history = skip_special_tokens_in_history
        self.progress_bar = progress_bar
        # Set by the loop so ``decode`` can report the generated region.
        self._last_prompt_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Protocol helpers
    # ------------------------------------------------------------------
    def reset(self):
        pass

    @property
    def train_insertion_schedule(self) -> Schedule:
        assert self.noise_schedule is not None, "noise_schedule is required"
        return self.noise_schedule.insertion_noise_schedule

    @property
    def unmasking_schedule(self) -> Schedule:
        assert self.noise_schedule is not None, "noise_schedule is required"
        return self.noise_schedule.unmasking_noise_schedule

    @property
    def insertion_schedule(self) -> Schedule:
        """Inference insertion schedule (falls back to the training schedule)."""
        if self.inference_insertion_schedule is not None:
            return self.inference_insertion_schedule
        return self.train_insertion_schedule

    def decode(self, results: Dict[str, Any]) -> Tuple[List[str], Integer[TT, " batch seq_len"]]:
        x: Integer[TT, " batch seq_len"] = results["x_t"]
        out: List[str] = self.tokenizer.batch_decode(  # type: ignore[union-attr]
            x, skip_special_tokens=self.skip_special_tokens_in_history
        )
        return out, x

    def decode_generated(
        self,
        ids: Integer[TT, " batch seq_len"],
        attention_mask: Bool[TT, " batch seq_len"],
        prompt_mask: Bool[TT, " batch seq_len"],
    ) -> List[str]:
        """Decode the generated region (attended, non-prompt tokens), special tokens skipped."""
        texts: List[str] = []
        keep = attention_mask.bool() & ~prompt_mask.bool()
        for row_ids, row_keep in zip(ids.detach().cpu(), keep.detach().cpu()):
            gen_ids = row_ids[row_keep].tolist()
            texts.append(
                self.tokenizer.decode(gen_ids, skip_special_tokens=True)  # type: ignore[union-attr]
            )
        return texts

    def to_dict(
        self,
        batch: Dict[str, Any],
        preds: FlexMDMPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        history_data = preds.get("history", [[] for _ in range(len(preds["text"]))])
        formatted_history = self.format_history_for_output(history_data, round_precision=4)
        to_zip = [
            preds["text"],
            preds["generated_text"],
            preds["ids"].tolist(),
            formatted_history,
            preds.get("time_taken", cycle([-1])),
            preds.get("generated_length", cycle([-1])),
        ]
        dicts: List[Dict[str, Any]] = []
        for text, generated_text, ids, hist, time_taken, gen_len in zip(*to_zip):
            dicts.append(
                {
                    "text": text,
                    "generated_text": generated_text,
                    "ids": ids,
                    "history": hist,
                    "time_taken": time_taken,
                    "generated_length": gen_len,
                }
            )
        return dicts

    # ------------------------------------------------------------------
    # Sampling loop (port of ``flexmdm_generate``)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        *,
        steps: Optional[int] = None,
        trace: bool = False,
    ) -> Dict[str, Any]:
        """Run the FlexMDM sampler from fixed prompt tokens.

        Returns ``{"sequences", "attention_mask", "prompt_mask"}`` and, with
        ``trace=True``, ``history`` / ``attention_mask_history`` /
        ``insertion_history`` lists (step 0 = the prompt) exactly like the reference.
        """
        assert self.model is not None, "model is required"
        assert self.tokenizer is not None, "tokenizer is required"
        steps = int(self.max_steps if steps is None else steps)
        mask_id = int(self.tokenizer.mask_token_id)
        pad_id = int(self.tokenizer.pad_token_id)
        eps = self.eps
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape (batch, length), got {tuple(input_ids.shape)}."
            )
        if input_ids.eq(mask_id).any():
            raise ValueError("input_ids should not contain mask tokens initially.")

        device = input_ids.device
        xt = input_ids.clone()
        batch_size = xt.shape[0]
        dt = 1.0 / steps
        t = torch.zeros(batch_size, device=device, dtype=torch.float32)
        insertion_carry = torch.zeros_like(xt, dtype=torch.float32)
        if attention_mask is None:
            xt_attention_mask = infer_prompt_attention_mask(xt, pad_id=pad_id)
        else:
            xt_attention_mask = attention_mask.to(device=device).bool()
        if prompt_mask is None:
            prompt_mask = xt_attention_mask.clone()
        else:
            prompt_mask = prompt_mask.to(device=device).bool()
        if (prompt_mask & ~xt_attention_mask).any():
            raise ValueError("prompt_mask must be a subset of attention_mask")
        prompt_len = prompt_mask.sum(dim=1)

        history: List[torch.Tensor] = []
        attention_history: List[torch.Tensor] = []
        insertion_history: List[torch.Tensor] = []
        last_is_inserted = torch.zeros_like(xt_attention_mask, dtype=torch.bool)
        text_history = self.create_history(batch_size)

        def record(step_done: int) -> None:
            nonlocal text_history
            if trace:
                history.append(xt.detach().clone())
                attention_history.append(xt_attention_mask.detach().clone())
                insertion_history.append(last_is_inserted.detach().clone())
            if self.return_history:
                text_history = self.update_history_from_state(
                    text_history, {"x_t": xt}, step_done
                )

        record(0)

        iterator = range(steps)
        if self.progress_bar:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="FlexMDM sampling")
            except ImportError:  # pragma: no cover
                pass

        train_ins = self.train_insertion_schedule
        infer_ins = self.insertion_schedule
        unmask_sched = self.unmasking_schedule
        for step_idx in iterator:
            # Schedule reparameterization: model sees the training-time coordinate
            # whose alpha equals the inference insertion schedule's alpha at t.
            if infer_ins is train_ins:
                model_t = t
            else:
                model_t = reparameterize_time(
                    t, inference_schedule=infer_ins, train_schedule=train_ins
                )
            logits, log_length_pred = self.model(xt, model_t, xt_attention_mask)

            # Dream-style shift: position i predicts token i + 1.
            unmask_logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            mask_indices = xt.eq(mask_id) & xt_attention_mask

            if step_idx == steps - 1:
                final_tokens = unmask_logits.argmax(dim=-1)
                xt = xt.clone()
                xt[mask_indices] = final_tokens[mask_indices]
                last_is_inserted = torch.zeros_like(xt_attention_mask, dtype=torch.bool)
                record(steps)
                break

            hazard_t = t + 0.5 * dt if self.use_midpoint_hazard_rate else t
            unmask_rate = schedule_hazard_rate(unmask_sched, hazard_t, eps)
            if self.confidence == "origin":
                xt = origin_unmask_step(
                    xt,
                    unmask_logits,
                    mask_indices,
                    rate=unmask_rate,
                    dt=dt,
                    temperature=self.temperature,
                )
            else:
                xt, _ = adaptive_unmask_step(
                    xt,
                    unmask_logits,
                    mask_indices,
                    rate=unmask_rate,
                    dt=dt,
                    temperature=self.temperature,
                    confidence_method=self.confidence,
                )

            insertion_rate = schedule_hazard_rate(infer_ins, hazard_t, eps)
            xt, xt_attention_mask, last_is_inserted, insertion_carry = insertion_step(
                xt,
                xt_attention_mask,
                log_length_pred,
                prompt_len=prompt_len,
                rate=insertion_rate,
                dt=dt,
                pad_id=pad_id,
                mask_id=mask_id,
                insertion_count_sampler=self.insertion_count_sampler,
                insertion_temperature=self.insertion_temperature,
                insertion_count_theta=self.insertion_count_theta,
                insertion_carry=insertion_carry,
                flush_carry=(step_idx == steps - 2),
            )

            t = (t + dt).clamp_max(1.0 - eps)
            record(step_idx + 1)

        out: Dict[str, Any] = {
            "sequences": xt,
            "attention_mask": xt_attention_mask,
            "prompt_mask": prompt_mask,
            "text_history": text_history,
        }
        if trace:
            out["history"] = history
            out["attention_mask_history"] = attention_history
            out["insertion_history"] = insertion_history
        return out

    @torch._dynamo.disable()
    def predict(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FlexMDMPredictionDict:
        start = time.time()
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        prompt_mask = batch.get("prompt_mask", batch.get("fixed"))
        out = self.sample(input_ids, attention_mask, prompt_mask)
        elapsed = time.time() - start

        ids = out["sequences"]
        attn = out["attention_mask"]
        pmask = out["prompt_mask"]
        batch_size = ids.shape[0]
        text = self.tokenizer.batch_decode(  # type: ignore[union-attr]
            [row[m].tolist() for row, m in zip(ids.cpu(), attn.cpu())],
            skip_special_tokens=True,
        )
        generated_text = self.decode_generated(ids, attn, pmask)
        generated_length = (attn.bool() & ~pmask.bool()).sum(dim=1).tolist()
        return {  # type: ignore[return-value]
            "text": text,
            "generated_text": generated_text,
            "ids": ids,
            "attention_mask": attn,
            "generated_length": generated_length,
            "time_taken": [elapsed / batch_size] * batch_size,
            "history": out["text_history"],
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def encode_prompts(
        self, prompts: List[str], max_length: int, prepend_bos: bool = True
    ) -> Dict[str, torch.Tensor]:
        from .datamodule_flexmdm_dream import encode_flexmdm_prompt_batch

        return encode_flexmdm_prompt_batch(
            self.tokenizer, prompts, max_length=max_length, prepend_bos=prepend_bos
        )

    def generate(self, prompts: List[str], max_length: Optional[int] = None) -> List[str]:
        assert self.model is not None
        max_length = int(max_length or getattr(self.model, "max_length", 768))
        batch = self.encode_prompts(prompts, max_length=max_length)
        device = next(self.model.parameters()).device  # type: ignore[union-attr]
        batch = {k: v.to(device) for k, v in batch.items()}
        return self.predict(batch)["generated_text"]


__all__ = [
    "FlexMDMDreamPredictor",
    "schedule_hazard_rate",
    "gumbel_sample",
    "adaptive_unmask_step",
    "origin_unmask_step",
    "insertion_step",
    "apply_after_token_insertions",
    "valid_insertion_gaps",
    "cap_insertions_to_capacity",
    "expected_insertion_counts",
    "infer_prompt_attention_mask",
]
