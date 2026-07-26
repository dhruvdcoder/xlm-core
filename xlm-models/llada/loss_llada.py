"""LLaDA SFT and RELAY BPTT losses (no logits shift; mask-token CE only)."""

from typing import Any, Dict, Optional, cast

import torch
import torch.nn.functional as F

from xlm.datamodule import Tokenizer
from xlm.harness import Harness, LossFunction
from xlm.utils.nn import masked_mean

MLMBatch = Dict[str, Any]
MLMLossDict = Dict[str, Any]


class LLaDASFTLoss(LossFunction[MLMBatch, MLMLossDict]):
    """Masked CE on answer-span masks with optional 1/t weighting.

    Expects a collator that already applied absorbing MDM masks on the answer
    span (``input_ids`` contains ``mask_token_id``; ``target_ids`` holds clean
    tokens). Weight ``1/t`` uses the empirical per-example mask rate over
    answer-eligible positions when ``answer_mask`` is provided; otherwise the
    rate over all non-pad positions.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Tokenizer] = None,
        use_time_weighting: bool = True,
        min_t: float = 1e-3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.use_time_weighting = use_time_weighting
        self.min_t = min_t
        self.mask_token_id_tensor = None

    def configure(self, pl_module: Harness) -> None:
        self.mask_token_id_tensor = torch.tensor(
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )

    def __call__(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        return self.loss_fn(batch, batch_idx, dataloader_idx, dataloader_name)

    def _forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
        h_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model = self.model
        out = model(input_ids, attention_mask, positions, h_t=h_t)
        if isinstance(out, tuple):
            return out[0]
        return out

    def loss_fn(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        targets = batch["target_ids"].clone()
        assert targets is not None

        positions = (attention_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        positions = positions * attention_mask.long()

        logits = self._forward_logits(input_ids, attention_mask, positions)
        masked = input_ids == self.mask_token_id_tensor
        ignore = ~masked
        targets = targets.clone()
        targets[ignore] = -100

        if ignore.all():
            return {
                "loss": logits.sum() * 0.0,
            }

        ce = F.cross_entropy(
            logits.transpose(1, 2),
            targets,
            reduction="none",
            ignore_index=-100,
        )
        if self.use_time_weighting:
            answer_mask = batch.get("answer_mask")
            if answer_mask is not None:
                denom = answer_mask.to(dtype=torch.bool).sum(dim=-1).clamp(min=1)
            else:
                denom = attention_mask.sum(dim=-1).clamp(min=1)
            t = (masked.sum(dim=-1).to(dtype=ce.dtype) / denom.to(dtype=ce.dtype)).clamp(
                min=self.min_t
            )
            ce = ce * (1.0 / t).unsqueeze(-1)

        loss = masked_mean(ce.flatten(), masked.flatten(), dim=-1)
        return {"loss": loss + 0.0 * logits.sum()}


class LLaDARelayBPTTLoss(LLaDASFTLoss):
    """K-step on-policy RELAY unroll with optional stop-grad through ``h_s``.

    Step 1: forward with ``h_t=0`` on the collator-masked batch; CE on masks;
    teacher-force positions with confidence > ``threshold`` (plus per-example
    argmax fallback). Step 2: forward remaining masks with carried ``h_s``
    (detached when ``stop_grad_h_s``). Total loss is the mean of the step losses.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Tokenizer] = None,
        use_time_weighting: bool = False,
        min_t: float = 1e-3,
        num_steps: int = 2,
        stop_grad_h_s: bool = False,
        threshold: float = 0.85,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_time_weighting=use_time_weighting,
            min_t=min_t,
        )
        self.num_steps = int(num_steps)
        self.stop_grad_h_s = bool(stop_grad_h_s)
        self.threshold = float(threshold)

    def _compute_ce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masked: torch.Tensor,
    ) -> torch.Tensor:
        tgt = targets.clone()
        tgt[~masked] = -100
        if (~masked).all():
            return logits.sum() * 0.0
        ce = F.cross_entropy(
            logits.transpose(1, 2),
            tgt,
            reduction="none",
            ignore_index=-100,
        )
        return masked_mean(ce.flatten(), masked.flatten(), dim=-1)

    @torch.no_grad()
    def _select_unmask(
        self,
        logits: torch.Tensor,
        masked: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        conf, _ = probs.max(dim=-1)
        conf = torch.where(masked, conf, torch.full_like(conf, float("-inf")))
        unmask = (conf > self.threshold) & masked
        # Per-example argmax fallback so every non-empty mask row reveals ≥1 token.
        has_mask = masked.any(dim=-1)
        if has_mask.any():
            argmax_pos = conf.argmax(dim=-1)
            b_idx = torch.arange(logits.shape[0], device=logits.device)
            unmask[b_idx[has_mask], argmax_pos[has_mask]] = True
            unmask = unmask & masked
        return unmask

    def _forward_relay(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
        h_t: Optional[torch.Tensor],
    ):
        model = cast(Any, self.model)
        out = model(input_ids, attention_mask, positions, h_t=h_t)
        if isinstance(out, tuple):
            return out[0], out[1]
        # Vanilla model path: no carry.
        return out, None

    def loss_fn(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        x = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        targets = batch["target_ids"]
        assert targets is not None

        positions = (attention_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        positions = positions * attention_mask.long()

        d_model = int(getattr(self.model, "d_model", self.model.config.d_model))
        h = torch.zeros(
            x.shape[0],
            x.shape[1],
            d_model,
            device=x.device,
            dtype=torch.float32,
        )

        # Suffix-block collator can mask pad slots whose targets are -100; never
        # teacher-force those into input_ids (Embedding rejects negative indices).
        valid_target = targets.ne(-100)

        loss_terms = []
        for _step in range(self.num_steps):
            masked = (x == self.mask_token_id_tensor) & valid_target
            logits, h_s = self._forward_relay(x, attention_mask, positions, h_t=h)
            L_t = self._compute_ce(logits, targets, masked) + 0.0 * logits.sum()
            loss_terms.append(L_t)

            unmask = self._select_unmask(logits.detach(), masked)
            x = x.clone()
            x[unmask] = targets[unmask]

            if h_s is None:
                continue
            h_next = h_s.detach() if self.stop_grad_h_s else h_s
            if masked.any():
                h = h_next

        loss = torch.stack(loss_terms).mean()
        return {"loss": loss}
