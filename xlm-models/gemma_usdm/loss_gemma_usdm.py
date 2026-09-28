"""Canvas cross-entropy and the two-visit self-conditioning buffer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from gemma_usdm.noise_gemma_usdm import corrupt_canvas, sample_time
from gemma_usdm.types_gemma_usdm import StoredRow
from xlm.datamodule import Tokenizer
from xlm.harness import Harness
from xlm.utils.nn import masked_mean
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class StreamingSelfCondBuffer:
    """One replay per example, then eviction.

    Capacity is the fresh loader batch. Visit 1 stores the noised tokens and
    the detached soft embedding ``p_hat E``. The next step replays that row
    and drops it. Nothing is written back after the second visit.
    """

    def __init__(self) -> None:
        self.rows: Optional[StoredRow] = None

    def reset(self) -> None:
        self.rows = None

    def pop_compatible(self, shape: torch.Size, device: torch.device) -> Optional[StoredRow]:
        rows = self.rows
        self.rows = None
        if rows is None:
            return None
        stored = rows["input_ids"]
        if stored.shape != shape or stored.device != device:
            return None
        return rows

    def store(self, rows: StoredRow) -> None:
        self.rows = {
            "input_ids": rows["input_ids"].detach().clone(),
            "attention_mask": rows["attention_mask"].detach().clone(),
            "target_ids": rows["target_ids"].detach().clone(),
            "canvas_mask": rows["canvas_mask"].detach().clone(),
            "soft": rows["soft"].detach().clone(),
        }


class GemmaUsdmLoss:
    """One forward per step. The fresh half has ``z = 0``; the replay half does not.

    ``loss_on_padding`` is applied by the collator (pad targets are ``-100``
    when it is false). This loss supervises every target that is not ``-100``,
    including canvas tokens that were not replaced.
    """

    def __init__(
        self,
        loss_on_padding: bool = True,
        clean_prob: float = 0.1,
        model: Any = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.loss_on_padding = loss_on_padding
        self.clean_prob = clean_prob
        self.model = model
        self.tokenizer = tokenizer
        self.buffer = StreamingSelfCondBuffer()
        self._log_remaining = 4
        self._logged_replay = False
        self.last_fresh_rows = 0
        self.last_forward_rows = 0
        self.last_forward_input_ids: Optional[torch.Tensor] = None
        self.last_replay_soft: Optional[torch.Tensor] = None

    def configure(self, pl_module: Harness) -> None:
        del pl_module

    def _corrupt(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        clean = batch["input_ids"]
        canvas_mask = batch["canvas_mask"].to(dtype=torch.bool)
        vocab = self.model.embed_tokens.num_embeddings
        pad_id = int(self.model.padding_idx)
        t = sample_time(clean.shape[0], self.clean_prob, clean.device)
        noisy = corrupt_canvas(clean, canvas_mask, t, vocab, pad_id)
        return {
            "input_ids": noisy,
            "attention_mask": batch["attention_mask"].to(dtype=torch.bool),
            "target_ids": batch["target_ids"],
            "canvas_mask": canvas_mask,
        }

    def _soft_embedding(
        self, logits: torch.Tensor, canvas_mask: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        weight = self.model.embed_tokens.weight.detach().float()
        soft = probs @ weight
        return soft * canvas_mask.to(dtype=soft.dtype).unsqueeze(-1)

    def _maybe_log(self, fresh_rows: int, forward_rows: int) -> None:
        replay = fresh_rows > 0 and forward_rows == 2 * fresh_rows
        if self._log_remaining <= 0 and (not replay or self._logged_replay):
            return
        logger.info(
            "gemma_usdm fresh_rows=%d forward_rows=%d",
            fresh_rows,
            forward_rows,
        )
        if self._log_remaining > 0:
            self._log_remaining -= 1
        if replay:
            self._logged_replay = True

    def loss_fn(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
        self_cond: torch.Tensor,
    ):
        """Compiled forward and canvas cross-entropy. No buffer traffic."""
        attention_mask = attention_mask.to(dtype=torch.bool)
        positions = (attention_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        positions = positions * attention_mask.long()
        flash = bool(getattr(self.model, "force_flash_attn", False))
        logits = self.model(
            input_ids,
            None if flash else attention_mask,
            positions,
            self_cond=self_cond,
        )
        ce = F.cross_entropy(
            logits.transpose(1, 2),
            target_ids,
            reduction="none",
            ignore_index=-100,
        )
        valid = target_ids != -100
        loss = masked_mean(ce.flatten(), valid.flatten(), dim=-1)
        return loss, logits

    def __call__(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        del batch_idx, dataloader_idx, dataloader_name
        fresh = self._corrupt(batch)
        training = bool(self.model.training)
        carried: Optional[StoredRow] = None
        if training:
            carried = self.buffer.pop_compatible(
                fresh["input_ids"].shape, fresh["input_ids"].device
            )
        else:
            self.buffer.reset()

        n_fresh = fresh["input_ids"].shape[0]
        d_model = self.model.embed_tokens.embedding_dim
        weight_dtype = self.model.embed_tokens.weight.dtype
        device = fresh["input_ids"].device
        seq = fresh["input_ids"].shape[1]
        self.last_replay_soft = None

        if carried is None:
            assembled = fresh
            self_cond = torch.zeros(
                n_fresh, seq, d_model, device=device, dtype=weight_dtype
            )
        else:
            self.last_replay_soft = carried["soft"]
            z_carried = self.model.self_conditioning(
                carried["soft"], carried["canvas_mask"]
            )
            zeros = torch.zeros(
                n_fresh, seq, d_model, device=device, dtype=z_carried.dtype
            )
            self_cond = torch.cat([zeros, z_carried], dim=0)
            assembled = {
                key: torch.cat([fresh[key], carried[key]], dim=0)
                for key in (
                    "input_ids",
                    "attention_mask",
                    "target_ids",
                    "canvas_mask",
                )
            }

        self.last_fresh_rows = n_fresh
        self.last_forward_rows = assembled["input_ids"].shape[0]
        self.last_forward_input_ids = assembled["input_ids"]

        loss, logits = self.loss_fn(
            assembled["input_ids"],
            assembled["attention_mask"],
            assembled["target_ids"],
            self_cond,
        )
        if training:
            with torch.no_grad():
                soft = self._soft_embedding(logits[:n_fresh], fresh["canvas_mask"])
            self.buffer.store(
                {
                    "input_ids": fresh["input_ids"],
                    "attention_mask": fresh["attention_mask"],
                    "target_ids": fresh["target_ids"],
                    "canvas_mask": fresh["canvas_mask"],
                    "soft": soft,
                }
            )
            self._maybe_log(self.last_fresh_rows, self.last_forward_rows)
        return {"loss": loss}
