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
import torch.nn.functional as F
from jaxtyping import Bool, Integer, Float
from xlm.datamodule import Tokenizer
from torch import Tensor as TT
from .types_elm import (
    ELMBatch,
    ELMPredictionDict,
    ELMModel,
    ELMSeq2SeqPredictionBatch,
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

ELMStepResults = Dict[str, Any]
# class ELMStepResults(TypedDict):
#    """Step results for ELM.
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


class ELMPredictor(torch.nn.Module, Predictor[ELMBatch, ELMPredictionDict]):
    """Base predictor for ELM. Stochastically selects positions to unmask based on max_steps and max_new_tokens."""

    def __init__(
        self,
        max_steps: int,
        max_new_tokens: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[ELMModel] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """Initialize ELM Predictor.

        Args:
            max_steps: Maximum number of prediction steps.
            tokenizer: Tokenizer for encoding/decoding.
            noise_schedule: Noise schedule for the diffusion process.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            model: The ELM model to use for predictions.
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

    def decode(self, results: ELMStepResults) -> Tuple[
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
        step_results: ELMStepResults,
    ) -> Bool[TT, " batch"]:
        return step_results["done"].all().item()

    def predict_single_step(
        self,
        step_results: ELMStepResults,
        final_step: bool = False,
    ) -> ELMStepResults:
        # fmt: off
        threshold = 0.8
        attention_mask: Bool[TT, " batch seq_len"] = step_results["attention_mask"]
        x = step_results["x"]
        positions = step_results["positions"]
        output_start_idx = step_results["output_start_idx"]
        done = step_results["done"]
        k = step_results["k"] + 1
        # fmt: on
        # TODO (efficiency): Logits can be cached if nothing in the input changes
        assert self.model is not None, "Model is not initialized"
        logits = self.model(x, attention_mask, positions)
        probs = F.softmax(logits, dim=-1)
        max_probs, token_indices = torch.max(probs, dim=-1)
        suffix_max_probs = max_probs[:, output_start_idx:]
        topk_probs, topk_seq_indices = torch.topk(suffix_max_probs, k, dim=1)
        topk_seq_indices = topk_seq_indices + output_start_idx
        remove_mask = torch.zeros_like(x, dtype=torch.bool)
        remove_mask.scatter_(1, topk_seq_indices, True)
        x_updated = torch.full_like(x, self.tokenizer.mask_token_id)
        x_updated[remove_mask] = token_indices[remove_mask]
        ## x_updated = torch.where(max_probs >= threshold, token_indices, torch.full_like(token_indices, self.tokenizer.mask_token_id))
        x[:, output_start_idx:] = x_updated[:, output_start_idx:]
        # compute stopping condition
        if k == self.max_new_tokens:
            done = True
        return {
            "x": x,
            "attention_mask": attention_mask,
            "positions": positions,
            "output_start_idx": output_start_idx,
            "logits": logits,
            "done": done,
            "k": k,
        }

    def to_dict(
        self,
        batch: ELMBatch,  # type: ignore
        preds: ELMPredictionDict,
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
    
    def postprocess(
            self,
            x: Integer[TT, " batch seq_len"],
            prefix_len: int,
    ) -> Integer[TT, " batch seq_len"]:
        n, l = x.shape
        prefix = x[:, :prefix_len]
        suffix = x[:, prefix_len:]
        ## We treat pad token as normal token.
        ## suffix = torch.where(suffix == self.tokenizer.pad_token_id, torch.full_like(suffix, self.tokenizer.mask_token_id), suffix)
        key = (suffix == self.tokenizer.mask_token_id).long()
        sorted_indices = torch.argsort(key, dim=1, stable=True)
        batch_indices = torch.arange(n, device=x.device).unsqueeze(1).expand(-1, l - prefix_len)
        sorted_suffix =suffix[batch_indices, sorted_indices]
        out = torch.cat([prefix, sorted_suffix], dim=1)
        return out

    @torch._dynamo.disable()
    def predict(
        self,
        batch: ELMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ELMPredictionDict:
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

        step_results: ELMStepResults = {
            "x": x,
            "attention_mask": attention_mask,
            "positions": positions,
            "output_start_idx": output_start_idx,
            "logits": None,  # type: ignore # ok for first step
            "done": False,
            "k": 0
        }
        while not step_results["done"]:
            step_results = self.predict_single_step(
                step_results,
            )
            step_results["x"] = self.postprocess(step_results["x"], output_start_idx)
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
