"""LLaDA diffusion predictor with optional RELAY carry and NFE logging.

LLaDA's reference sampler fills a canvas of ``mask_token_id`` (126336) and
iteratively commits the most confident predictions per semi-AR block.
``DreamPredictor`` already implements that machinery; this subclass adds:

- ``with_relay``: carry ``h_s → h_t`` across denoising steps
- ``nfe`` / ``mean_nfe`` in prediction outputs (``steps_taken`` == NFE for LLaDA)

Unlike Dream, do NOT configure a ``logits_hook`` (no next-token shift).
"""

from typing import Any, Dict, List, Optional

import torch
from dream.predictor_dream import DreamPredictor
from jaxtyping import Bool
from torch import Tensor as TT

from mlm.types_mlm import MLMBatch, MLMPredictionDict
from xlm.utils.nn import select_random_indices

MLMStepResults = Dict[str, Any]


class LLaDAPredictor(DreamPredictor):
    """Masked-diffusion predictor for LLaDA with optional RELAY carry + NFE."""

    def __init__(self, *args, with_relay: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_relay = bool(with_relay)

    def predict_single_step(
        self,
        step_results: MLMStepResults,
        final_step: bool = False,
    ) -> MLMStepResults:
        if not self.with_relay:
            return super().predict_single_step(step_results, final_step=final_step)

        attention_mask: Bool[TT, " batch seq_len"] = step_results["attention_mask"]
        x = step_results["x"]
        positions = step_results["positions"]
        current_step = step_results["current_step"]
        input_end_positions = step_results["input_end_positions"]
        tokenizer = self._require_tokenizer()
        assert self.model is not None, "Model is not initialized"

        h_t = step_results.get("h")
        out = self.model(
            x,
            attention_mask if not self.flash_attn else None,
            positions,
            h_t=h_t,
        )
        if isinstance(out, tuple):
            logits, h_s = out
        else:
            logits, h_s = out, None

        if self.logits_hook is not None:
            logits = self.logits_hook(logits)

        masked = x == tokenizer.mask_token_id
        allowed = self.allowed(current_step, input_end_positions, x.shape[-1])
        _allowed = masked.logical_and(allowed)

        if final_step:
            unmask = _allowed
        else:
            num_unmask = self.max_new_tokens // self.max_steps
            if self.confidence is not None:
                score = self._compute_confidence(logits, _allowed)
                selection_mode = (
                    "greedy" if self.confidence_temperature == 0 else "sample"
                )
                unmask = select_random_indices(
                    inp_shape=x.shape,
                    num_unmask=torch.full(
                        (x.shape[0],),
                        num_unmask,
                        device=x.device,
                        dtype=torch.long,
                    ),
                    select_from_mask=_allowed,
                    selection_score=score,
                    selection_mode=selection_mode,
                    temperature=self.confidence_temperature,
                )
            else:
                unmask = select_random_indices(
                    inp_shape=x.shape,
                    num_unmask=torch.full(
                        (x.shape[0],),
                        num_unmask,
                        device=x.device,
                        dtype=torch.long,
                    ),
                    select_from_mask=_allowed,
                    selection_score=None,
                    selection_mode="sample",
                    temperature=self.confidence_temperature,
                )

        x = x.clone()
        x[unmask] = self.sampling_function(logits[unmask])

        result: MLMStepResults = {
            "x": x,
            "attention_mask": attention_mask,
            "positions": positions,
            "logits": logits,
            "current_step": current_step + 1,
            "input_end_positions": input_end_positions,
            "done": step_results.get(
                "done",
                torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
            ),
        }
        if h_s is not None:
            result["h"] = h_s.detach()
        return result

    def predict(
        self,
        batch: MLMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMPredictionDict:
        preds = super().predict(
            batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            dataloader_name=dataloader_name,
        )
        steps = preds.get("steps_taken")
        if steps is not None:
            preds["nfe"] = list(steps)
            preds["mean_nfe"] = float(sum(steps) / max(len(steps), 1))
        return preds

    def to_dict(
        self,
        batch: MLMBatch,  # type: ignore
        preds: MLMPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        dicts = super().to_dict(
            batch,
            preds,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            dataloader_name=dataloader_name,
        )
        nfe = preds.get("nfe")
        if nfe is not None:
            for i, row in enumerate(dicts):
                if i < len(nfe):
                    row["nfe"] = nfe[i]
        return dicts
