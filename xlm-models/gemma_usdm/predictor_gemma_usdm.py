"""Entropy-bounded sampler (DiffusionGemma Algorithm 1) with per-sequence NFE."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from gemma_usdm.noise_gemma_usdm import sample_uniform_except_pad
from xlm.datamodule import Tokenizer


def entropy_bounded_commit(
    tokens: Tensor,
    entropy: Tensor,
    canvas_mask: Tensor,
    sampled: Tensor,
    uniform: Tensor,
    budget: float,
) -> Tuple[Tensor, Tensor]:
    """Commit the lowest-entropy canvas positions whose preceding sum is ``<= budget``.

    The first canvas position in that order is always eligible, because the
    sum before it is 0. Every other canvas position is replaced by ``uniform``.
    Ties in entropy keep the lower index first.
    """
    canvas_mask = canvas_mask.to(dtype=torch.bool)
    sort_key = entropy.masked_fill(~canvas_mask, float("inf"))
    order = torch.argsort(sort_key, dim=-1, stable=True)
    sorted_entropy = entropy.gather(1, order)
    sorted_canvas = canvas_mask.gather(1, order)
    preceding = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy
    commit_sorted = sorted_canvas & (preceding <= budget)
    commit = torch.zeros_like(canvas_mask)
    commit.scatter_(1, order, commit_sorted)
    updated = torch.where(commit, sampled, tokens)
    renoise = canvas_mask & ~commit
    updated = torch.where(renoise, uniform, updated)
    return updated, commit


def tempered_probs(logits: Tensor, tau: float) -> Tuple[Tensor, Tensor]:
    """Softmax at temperature ``tau`` and its entropy."""
    log_p = torch.log_softmax(logits.float() / tau, dim=-1)
    probs = log_p.exp()
    entropy = -(probs * log_p).sum(dim=-1)
    return probs, entropy


def canvas_stop_mask(
    argmax: Tensor,
    canvas_mask: Tensor,
    loss_on_padding: bool,
    eos_id: int,
) -> Tensor:
    """Positions that count toward the stop rule.

    ``loss_on_padding=true`` uses the whole canvas. ``false`` uses positions
    through each row's first argmax EOS, EOS included. A canvas with no EOS
    uses the whole canvas.
    """
    canvas_mask = canvas_mask.to(dtype=torch.bool)
    if loss_on_padding:
        return canvas_mask
    is_eos = (argmax == eos_id) & canvas_mask
    has_eos = is_eos.any(dim=-1)
    first = is_eos.int().argmax(dim=-1)
    index = torch.arange(argmax.shape[-1], device=argmax.device).unsqueeze(0)
    through = canvas_mask & (index <= first.unsqueeze(-1))
    return torch.where(has_eos.unsqueeze(-1), through, canvas_mask)


def _decode_through_eos(
    canvas_ids: Tensor,
    eos_id: int,
    tokenizer: Tokenizer,
    skip_special_tokens: bool,
) -> List[str]:
    texts: List[str] = []
    for row in canvas_ids.tolist():
        if eos_id in row:
            row = row[: row.index(eos_id)]
        texts.append(
            tokenizer.decode(row, skip_special_tokens=skip_special_tokens)
        )
    return texts


class GemmaUsdmPredictor:
    """Algorithm 1 on the concatenated source and canvas. No encoder prefill."""

    def __init__(
        self,
        max_steps: int = 32,
        max_new_tokens: int = 128,
        entropy_budget: float = 0.1,
        stop_entropy: float = 0.005,
        tau_max: float = 0.8,
        tau_min: float = 0.4,
        loss_on_padding: bool = True,
        skip_special_tokens: bool = True,
        tokenizer: Optional[Tokenizer] = None,
        model: Any = None,
        noise_schedule: Any = None,
    ):
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.entropy_budget = entropy_budget
        self.stop_entropy = stop_entropy
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.loss_on_padding = loss_on_padding
        self.skip_special_tokens = skip_special_tokens
        self.tokenizer = tokenizer
        self.model = model
        self.noise_schedule = noise_schedule

    def _temperature(self, step: int) -> float:
        """Anneal ``tau_max`` at step 1 down to ``tau_min`` at the last step."""
        span = max(self.max_steps - 1, 1)
        t = 1.0 - (step - 1) / span
        return self.tau_min + (self.tau_max - self.tau_min) * t

    @torch.no_grad()
    def predict(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        del batch_idx, dataloader_idx, dataloader_name
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("GemmaUsdmPredictor.tokenizer is not set")
        model = self.model
        if model is None:
            raise RuntimeError("GemmaUsdmPredictor.model is not set")

        started = time.time()
        prefix = batch["input_ids"]
        attention = batch.get("attention_mask")
        if attention is None:
            attention = torch.ones(
                prefix.shape, dtype=torch.bool, device=prefix.device
            )
        else:
            attention = attention.to(device=prefix.device, dtype=torch.bool)

        output_start_idx = prefix.shape[-1]
        batch_size = prefix.shape[0]
        pad_id = int(tokenizer.pad_token_id)
        eos_id = int(tokenizer.eos_token_id)
        vocab = int(model.embed_tokens.num_embeddings)
        canvas = sample_uniform_except_pad(
            (batch_size, self.max_new_tokens),
            vocab,
            pad_id,
            prefix.device,
        )
        tokens = torch.cat([prefix, canvas], dim=-1)
        canvas_mask = torch.zeros(
            tokens.shape, dtype=torch.bool, device=tokens.device
        )
        canvas_mask[:, output_start_idx:] = True
        attention = torch.cat(
            [
                attention,
                torch.ones(
                    (batch_size, self.max_new_tokens),
                    dtype=torch.bool,
                    device=tokens.device,
                ),
            ],
            dim=-1,
        )
        d_model = model.embed_tokens.embedding_dim
        z = torch.zeros(
            batch_size,
            tokens.shape[-1],
            d_model,
            device=tokens.device,
            dtype=model.embed_tokens.weight.dtype,
        )
        active = torch.ones(batch_size, dtype=torch.bool, device=tokens.device)
        nfe = torch.full(
            (batch_size,),
            self.max_steps,
            dtype=torch.long,
            device=tokens.device,
        )
        result = tokens.clone()
        prev_argmax: Optional[Tensor] = None

        for step in range(1, self.max_steps + 1):
            if not bool(active.any()):
                break
            positions = (attention.long().cumsum(dim=1) - 1).clamp(min=0)
            positions = positions * attention.long()
            logits = model(
                tokens,
                attention,
                positions,
                self_cond=z,
            )
            argmax = logits.argmax(dim=-1)
            tau = self._temperature(step)
            probs, entropy = tempered_probs(logits, tau)
            region = canvas_stop_mask(
                argmax, canvas_mask, self.loss_on_padding, eos_id
            )
            counts = region.sum(dim=-1)
            mean_entropy = (entropy * region.float()).sum(dim=-1) / counts.clamp(
                min=1
            ).float()
            if prev_argmax is None:
                stable = torch.zeros_like(active)
            else:
                stable = ((argmax == prev_argmax) | ~region).all(dim=-1)
            entropy_ok = (counts > 0) & (mean_entropy <= self.stop_entropy)
            just_stopped = active & entropy_ok & stable & (step > 1)
            if step == self.max_steps:
                just_stopped = active
            canvas_argmax = torch.where(canvas_mask, argmax, tokens)
            result = torch.where(just_stopped.unsqueeze(-1), canvas_argmax, result)
            nfe = torch.where(
                just_stopped,
                torch.full_like(nfe, step),
                nfe,
            )
            active = active & ~just_stopped
            if (not bool(active.any())) or step == self.max_steps:
                break

            sampled = torch.multinomial(
                probs.reshape(-1, vocab), 1
            ).view(tokens.shape)
            uniform = sample_uniform_except_pad(
                tokens.shape, vocab, pad_id, tokens.device
            )
            proposed, _commit = entropy_bounded_commit(
                tokens,
                entropy,
                canvas_mask,
                sampled,
                uniform,
                self.entropy_budget,
            )
            soft = probs.float() @ model.embed_tokens.weight.detach().float()
            z_proposed = model.self_conditioning(soft, canvas_mask)
            keep = active.unsqueeze(-1)
            tokens = torch.where(keep, proposed, tokens)
            z = torch.where(keep.unsqueeze(-1), z_proposed, z)
            prev_argmax = argmax

        elapsed = time.time() - started
        nfe_list = [int(v) for v in nfe.detach().cpu().tolist()]
        return {
            "ids": result,
            "output_start_idx": output_start_idx,
            "steps_taken": nfe,
            "nfe": nfe_list,
            "time_taken": [elapsed] * batch_size,
            "loss": None,
        }

    def to_dict(
        self,
        batch: Dict[str, Any],
        preds: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        del batch, batch_idx, dataloader_idx, dataloader_name
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("GemmaUsdmPredictor.tokenizer is not set")
        start = int(preds["output_start_idx"])
        canvas = preds["ids"][:, start:]
        texts = _decode_through_eos(
            canvas,
            int(tokenizer.eos_token_id),
            tokenizer,
            self.skip_special_tokens,
        )
        nfe = preds["nfe"]
        if torch.is_tensor(nfe):
            nfe = [int(v) for v in nfe.detach().cpu().tolist()]
        rows: List[Dict[str, Any]] = []
        ids = preds["ids"]
        times = preds.get("time_taken", [-1] * len(texts))
        for i, text in enumerate(texts):
            rows.append(
                {
                    "text": text,
                    "generated_text": text,
                    "ids": ids[i].detach().cpu().tolist(),
                    "nfe": int(nfe[i]),
                    "steps_taken": int(nfe[i]),
                    "time_taken": times[i],
                    "output_start_idx": start,
                }
            )
        return rows

    def generate(self, prompts: List[str]) -> List[str]:
        raise NotImplementedError(
            "gemma-usdm generates through the prediction dataloader."
        )
