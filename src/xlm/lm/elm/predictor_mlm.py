from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)
from functools import partial
import torch
from jaxtyping import Bool, Integer, Float
from xlm.datamodule import Tokenizer
from torch import Tensor as TT
from .types_mlm import (
    MLMBatch,
    MLMPredictionDict,
    MLMModel,
    MLMSeq2SeqPredictionBatch,
)
from xlm.harness import Predictor
from xlm.noise import NoiseSchedule
from xlm.utils.nn import (
    sample_from_logits,
    sample_from_top_k,
    sample_from_top_p,
    sample_categorical,
    select_random_indices,
)
import time

MLMStepResults = Dict[str, Any]
# class MLMStepResults(TypedDict):
#    """Step results for MLM.
#
#    Attributes:
#        x: Integer[TT, " batch seq_len"] Current predicted sequence.
#        attention_mask: Bool[TT, " batch seq_len"] Mask of the current sequence.
#        logits: Float[TT, " batch seq_len vocab_size"] Logits of the current sequence.
#        done: Bool[TT, " batch"] Whether the current sequence is done.
#        steps_taken: Integer[TT, " batch"] Number of steps taken.
#        change: Bool[TT, " batch"] Whether any token in the current sequence is changed.
#        constraint: Bool[TT, " batch seq_len"] Constraint of the current sequence.
#        positions: Integer[TT, " batch seq_len"] Positions of the current sequence.
#    """
#
#    x: Integer[TT, " batch seq_len"]
#    attention_mask: Bool[TT, " batch seq_len"]
#    logits: Float[TT, " batch seq_len vocab_size"]
#    done: Bool[TT, " batch"]
#    steps_taken: Integer[TT, " batch"]
#    change: Optional[Bool[TT, " batch"]]
#    constraint: Optional[Bool[TT, " batch seq_len"]]
#    positions: Integer[TT, " batch seq_len"]


class MLMPredictor(torch.nn.Module, Predictor[MLMBatch, MLMPredictionDict]):
    """Base predictor for MLM. Stochastically selects positions to unmask based on max_steps and max_new_tokens."""

    def __init__(
        self,
        max_steps: int,
        max_new_tokens: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[MLMModel] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """Initialize MLM Predictor.

        Args:
            max_steps: Maximum number of prediction steps.
            tokenizer: Tokenizer for encoding/decoding.
            noise_schedule: Noise schedule for the diffusion process.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            model: The MLM model to use for predictions.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required")

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        if top_k is not None and top_p is not None:
            self.sampling_function = sample_from_logits
        elif top_k is not None and top_p is None:
            self.sampling_function = partial(sample_from_top_k, top_k)
        elif top_k is None and top_p is not None:
            self.sampling_function = partial(sample_from_top_p, top_p)
        else:
            raise ValueError("Both top_k and top_p cannot be non-None")

        self.noise_schedule = noise_schedule

    def reset(self):
        # simple predictor has no state
        pass

    def decode(self, results: MLMStepResults) -> Tuple[
        List[str],
        Integer[TT, " batch seq_len"],
    ]:
        """
        Args:
            results:
                x: Integer[TT, " batch seq_len"] Current predicted sequence.
        Returns:
            out: List[str] Decoded sequence with special tokens.
            x: Integer[TT, " batch seq_len"] Current predicted sequence.
        """
        x: Integer[TT, " batch seq_len"] = results["x"]
        out_with_spl_tokens: List[str] = self.tokenizer.batch_decode(
            x, skip_special_tokens=False
        )
        return out_with_spl_tokens, x

    def stop(
        self,
        step_results: MLMStepResults,
    ) -> Bool[TT, " batch"]:
        return step_results["done"]

    def predict_single_step(
        self,
        step_results: MLMStepResults,
        final_step: bool = False,
    ) -> MLMStepResults:
        # fmt: off
        attention_mask: Bool[TT, " batch seq_len"] = step_results["attention_mask"]
        x = step_results["x"]
        positions = step_results["positions"]
        # fmt: on
        # TODO (efficiency): Logits can be cached if nothing in the input changes
        assert self.model is not None, "Model is not initialized"
        logits = self.model(x, attention_mask, positions)
        masked = x == self.tokenizer.mask_token_id
        steps_left = self.max_steps - step_results["steps_taken"]
        if not final_step:
            num_unmask = (
                masked.sum(dim=-1) / (steps_left).clamp(min=1)
            ).long()
            unmask = select_random_indices(
                inp_shape=x.shape,
                num_unmask=num_unmask,
                select_from_mask=masked,
                selection_score=None,  # uniform
                selection_mode="sample",
            )
        else:
            unmask = masked
        x[unmask] = self.sampling_function(logits[unmask])
        # compute stopping condition
        step_results["steps_taken"] += 1
        done = step_results["steps_taken"] >= self.max_steps
        return {
            "x": x,
            "attention_mask": attention_mask,
            "positions": positions,
            "logits": logits,
            "steps_taken": step_results["steps_taken"],
            "done": done,
        }

    def to_dict(
        self,
        batch: MLMBatch,  # type: ignore
        preds: MLMPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        dicts: List[Dict[str, Any]] = []
        for text, ids in zip(
            preds["text"],
            preds["ids"].tolist(),
        ):
            dicts.append(
                {
                    "text": text,
                    "ids": ids,
                }
            )
        return dicts

    @torch._dynamo.disable()
    def predict(
        self,
        batch: MLMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMPredictionDict:
        _start_time = time.time()
        x = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"]

        output_start_idx = x.shape[-1]
        if self.max_new_tokens is not None:
            # we assume prefix is left-padded
            x = torch.cat(
                [
                    x,
                    torch.full(
                        (x.shape[0], self.max_new_tokens),
                        self.tokenizer.mask_token_id,
                        device=x.device,
                    ),
                ],
                dim=-1,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], self.max_new_tokens),
                        device=attention_mask.device,
                    ),
                ],
                dim=-1,
            )
        positions = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0).long()

        step_results: MLMStepResults = {
            "x": x,
            "attention_mask": attention_mask,
            "positions": positions,
            "logits": None,  # type: ignore # ok for first step
            "steps_taken": torch.zeros(
                x.shape[0], dtype=torch.int, device=x.device
            ),
            "done": torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
        }
        while not self.stop(step_results):
            step_results = self.predict_single_step(
                step_results,
            )
        step_results = self.predict_single_step(
            step_results,
            final_step=True,
        )
        # decode the final step
        (
            out,
            final_x,
        ) = self.decode(step_results)

        _end_time = time.time()
        _time_taken = _end_time - _start_time
        self.reset()
        return {
            "text": out,
            "ids": final_x,
            "loss": None,
            "time_taken": [_time_taken]
            * len(out),  # cannot separate time for each sample
            "output_start_idx": output_start_idx,
        }
