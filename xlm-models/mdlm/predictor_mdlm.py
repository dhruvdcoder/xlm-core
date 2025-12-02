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
from .types_mdlm import (
    MDLMBatch,
    MDLMPredictionDict,
    MDLMModel,
    MDLMSeq2SeqPredictionBatch,
)
from xlm.harness import Predictor
from xlm.noise import NoiseSchedule
from xlm.utils.nn import (
    sample_from_top_k,
    sample_from_top_p,
    sample_categorical,
)
import time

MDLMStepResults = Dict[str, Any]


class MDLMPredictor(torch.nn.Module, Predictor[MDLMBatch, MDLMPredictionDict]):
    """Base predictor for MLM. Stochastically selects positions to unmask based on max_steps and max_new_tokens."""

    def __init__(
        self,
        max_steps: int,
        max_new_tokens: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[MDLMModel] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """Initialize MDLM Predictor.

        Args:
            max_steps: Maximum number of prediction steps.
            tokenizer: Tokenizer for encoding/decoding.
            noise_schedule: Noise schedule for the diffusion process.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            model: The MDLM model to use for predictions.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required")

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.dt = (1 - 1e-5) / (max_steps + 1)
        if top_k is not None and top_p is not None:
            self.sampling_function = sample_categorical
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

    def decode(self, results: MDLMStepResults) -> Tuple[
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
        x: Integer[TT, " batch seq_len"] = results["x_t"]
        out_with_spl_tokens: List[str] = self.tokenizer.batch_decode(
            x, skip_special_tokens=True
        )
        return out_with_spl_tokens, x

    def stop(
        self,
        step_results: MDLMStepResults,
    ) -> Bool[TT, " batch"]:
        time_ended = not bool((step_results["t"] > 0).any())
        all_filled = not (
            (step_results["x_t"] == self.tokenizer.mask_token_id).any()
        )
        return step_results["done"].all().item() or time_ended or all_filled

    def predict_single_step(
        self,
        step_results: MDLMStepResults,
        final_step: bool = False,
    ) -> MDLMStepResults:
        # fmt: off
        attention_mask: Bool[TT, " batch seq_len"] = step_results["attention_mask"]
        x_t = step_results["x_t"]
        positions = step_results["positions"]
        t = step_results["t"]
        # fmt: on
        s = t - self.dt
        dot_sigma_t: Float[TT, " batch"] = self.noise_schedule(t)[1]
        dot_sigma_s: Float[TT, " batch"] = self.noise_schedule(s)[1]
        assert self.model is not None, "Model is not initialized"
        logits = self.model(x_t, dot_sigma_t, attention_mask, positions)
        chance_s = -torch.expm1(-dot_sigma_s)  # 1 - exp(-dot_sigma_s)
        chance_t = -torch.expm1(-dot_sigma_t)  # 1 - exp(-dot_sigma_t)
        if not final_step:
            q_xs = torch.softmax(logits, dim=-1) * (
                (chance_t - chance_s)[:, None, None]
            )  # (*batch, seq_len, vocab_size)
            assert (q_xs >= 0).all()
            # predicting mask tokens
            q_xs[:, :, self.tokenizer.mask_token_id] = chance_s[:, None]
            x_s = self.sampling_function(q_xs)  # (*batch, seq_len)
        else:
            x_s = torch.argmax(logits, dim=-1)  # (*batch, seq_len)

        masked = x_t == self.tokenizer.mask_token_id
        # copy the input for input positions that were non-mask
        x_s = torch.where(masked, x_s, x_t)
        # compute stopping condition
        step_results["steps_taken"] += 1
        done = step_results["steps_taken"] >= self.max_steps
        return {
            "x_t": x_s,
            "attention_mask": attention_mask,
            "positions": positions,
            "logits": logits,
            "t": s,
            "steps_taken": step_results["steps_taken"],
            "done": done,
        }

    def to_dict(
        self,
        batch: MDLMBatch,  # type: ignore
        preds: MDLMPredictionDict,
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
        batch: MDLMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MDLMPredictionDict:
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

        step_results: MDLMStepResults = {
            "x_t": x,
            "attention_mask": attention_mask,
            "positions": positions,
            "logits": None,  # type: ignore # ok for first step
            "steps_taken": torch.zeros(
                x.shape[0], dtype=torch.int, device=x.device
            ),
            "done": torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
            "t": (
                torch.ones(x.shape[0], dtype=torch.float, device=x.device)
                if "t" not in batch
                else batch["t"]
            ),
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
            * len(
                out
            ),  # cannot separate time for each sample, so run with batch_size=1 for time analysis
            "output_start_idx": output_start_idx,
        }
