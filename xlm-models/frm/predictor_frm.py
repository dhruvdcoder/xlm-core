"""Euler sampler for Flow Reasoning Models with self-conditioning and clue clamp."""

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from xlm.datamodule import Tokenizer
from xlm.harness import Predictor
from xlm.noise import NoiseSchedule
from xlm.utils.text import remove_trailing_pads

from .flow_frm import apply_clamp, one_hot_tokens, simplex_noise, velocity
from .types_frm import FRMBatch, FRMModel, FRMPredictionDict
from .unbatch import unbatch


class FRMPredictor(torch.nn.Module, Predictor[FRMBatch, FRMPredictionDict]):
    def __init__(
        self,
        max_steps: int,
        max_new_tokens: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[FRMModel] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        k_inner: int = 1,
        skip_special_tokens: bool = True,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self.model = model
        self.noise_schedule = noise_schedule
        self.k_inner = max(int(k_inner), 1)
        self.skip_special_tokens = skip_special_tokens

    def _require_tokenizer(self) -> Tokenizer:
        if self.tokenizer is None:
            raise RuntimeError("FRMPredictor.tokenizer is not set")
        return self.tokenizer

    def to_dict(
        self,
        batch: FRMBatch,
        preds: FRMPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if "generated_text" not in preds and "output_start_idx" in preds:
            tokenizer = self._require_tokenizer()
            start = int(preds["output_start_idx"])
            preds = {
                **preds,
                "generated_text": tokenizer.batch_decode(
                    preds["ids"][:, start:],
                    skip_special_tokens=self.skip_special_tokens,
                ),
            }
        return unbatch(preds, length=len(preds["text"]))

    @torch._dynamo.disable()
    def predict(
        self,
        batch: FRMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FRMPredictionDict:
        tokenizer = self._require_tokenizer()
        model = self.model
        assert model is not None
        start_time = time.time()

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape, dtype=torch.bool, device=input_ids.device
            )
        else:
            attention_mask = attention_mask.to(
                device=input_ids.device, dtype=torch.bool
            )

        vocab_size = model.num_embeddings
        clamp_mask = batch.get("clamp_mask")
        if clamp_mask is None:
            clamp_mask = input_ids != tokenizer.mask_token_id
        clamp_mask = clamp_mask.to(dtype=torch.bool)

        clue_ids = batch.get("clue_ids")
        if clue_ids is None:
            clue_ids = input_ids.masked_fill(~clamp_mask, tokenizer.pad_token_id)

        output_start_idx = input_ids.shape[-1]
        if self.max_new_tokens is not None:
            extra_len = int(self.max_new_tokens)
            pad = torch.full(
                (input_ids.shape[0], extra_len),
                tokenizer.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, pad], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        input_ids.shape[0],
                        extra_len,
                        dtype=torch.bool,
                        device=input_ids.device,
                    ),
                ],
                dim=1,
            )
            clamp_mask = torch.cat(
                [
                    clamp_mask,
                    torch.zeros(
                        clamp_mask.shape[0],
                        extra_len,
                        dtype=torch.bool,
                        device=clamp_mask.device,
                    ),
                ],
                dim=1,
            )
            clue_ids = torch.cat(
                [
                    clue_ids,
                    torch.full(
                        (clue_ids.shape[0], extra_len),
                        tokenizer.pad_token_id,
                        dtype=clue_ids.dtype,
                        device=clue_ids.device,
                    ),
                ],
                dim=1,
            )
        else:
            output_start_idx = 0

        clue_onehot = one_hot_tokens(clue_ids, vocab_size)
        x = simplex_noise(
            torch.empty(
                input_ids.shape[0],
                input_ids.shape[1],
                vocab_size,
                device=input_ids.device,
                dtype=torch.float32,
            )
        )
        x = apply_clamp(x, clue_onehot, clamp_mask)
        positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)

        dt = 1.0 / max(int(self.max_steps), 1)
        s = None
        for step in range(int(self.max_steps)):
            t_val = step * dt
            t = torch.full(
                (input_ids.shape[0],),
                t_val,
                device=input_ids.device,
                dtype=torch.float32,
            )
            for _ in range(self.k_inner):
                logits = model(
                    x,
                    t,
                    attention_mask=attention_mask,
                    positions=positions,
                    s=s,
                    clue_ids=clue_ids,
                    clamp_mask=clamp_mask,
                )
                s = F.softmax(logits, dim=-1)
            assert s is not None
            v = velocity(s, x, t)
            x = x + v * dt
            x = apply_clamp(x, clue_onehot, clamp_mask)

        ids = s.argmax(dim=-1)
        ids = torch.where(clamp_mask, clue_ids, ids)

        decoded = tokenizer.batch_decode(
            ids, skip_special_tokens=self.skip_special_tokens
        )
        decoded = [
            remove_trailing_pads(text, tokenizer) for text in decoded
        ]
        elapsed = time.time() - start_time
        steps = torch.full(
            (ids.shape[0],),
            int(self.max_steps),
            dtype=torch.long,
            device=ids.device,
        )
        return FRMPredictionDict(
            loss=None,
            text=decoded,
            ids=ids,
            time_taken=[elapsed] * ids.shape[0],
            output_start_idx=output_start_idx,
            steps_taken=steps,
        )
