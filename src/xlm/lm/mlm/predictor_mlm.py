from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
)
from functools import partial
import torch
from jaxtyping import Bool, Integer, Float
from xlm.datamodule import Tokenizer
from torch import Tensor as TT
from .types_mlm import MLMBatch, MLMPredictionDict, MLMModel
from xlm.harness import Predictor
from xlm.noise import NoiseSchedule
from xlm.utils.nn import (
    sample_from_logits,
    sample_from_top_k,
    sample_from_top_p,
    sample_categorical,
)
import time


class MLMStepResults(TypedDict):
    """Step results for MLM.

    Attributes:
        x: Integer[TT, " batch seq_len"] Current predicted sequence.
        attention_mask: Bool[TT, " batch seq_len"] Mask of the current sequence.
        logits: Float[TT, " batch seq_len vocab_size"] Logits of the current sequence.
        done: Bool[TT, " batch"] Whether the current sequence is done.
        steps_taken: Integer[TT, " batch"] Number of steps taken.
        change: Bool[TT, " batch"] Whether any token in the current sequence is changed.
        constraint: Bool[TT, " batch seq_len"] Constraint of the current sequence.
        positions: Integer[TT, " batch seq_len"] Positions of the current sequence.
    """

    x: Integer[TT, " batch seq_len"]
    attention_mask: Bool[TT, " batch seq_len"]
    logits: Float[TT, " batch seq_len vocab_size"]
    done: Bool[TT, " batch"]
    steps_taken: Integer[TT, " batch"]
    change: Optional[Bool[TT, " batch"]]
    constraint: Optional[Bool[TT, " batch seq_len"]]
    positions: Integer[TT, " batch seq_len"]


class MLMPredictor(torch.nn.Module, Predictor[MLMBatch, MLMPredictionDict]):
    """Stochastically selects positions to unmask based on max_steps and max_new_tokens."""

    def __init__(
        self,
        max_steps: int,
        tokenizer: Optional[Tokenizer] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        model: Optional[MLMModel] = None,
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
        self.dt = (1 - 1e-5) / (max_steps + 1)
        if top_k is not None and top_p is not None:
            self.sampling_function = sample_from_logits
        elif top_k is not None and top_p is None:
            self.sampling_function = partial(sample_from_top_k, top_k)
        elif top_k is None and top_p is not None:
            self.sampling_function = partial(sample_from_top_p, top_p)
        else:
            raise ValueError("Both top_k and top_p cannot be non-None")

        self.noise_schedule = noise_schedule
        # state
        self.max_new_tokens = None
        self.current_step = 0

    def reset(self):
        self.max_new_tokens = None
        self.current_step = 0

    def decode(self, results: MLMStepResults) -> Tuple[
        List[str],
        List[str],
        Integer[TT, " batch seq_len"],
    ]:
        """
        Args:
            results:
                x: Integer[TT, " batch seq_len"] Current predicted sequence.
        Returns:
            out: List[str] Decoded sequence.
            out_with_spl_tokens: List[str] Decoded sequence with special tokens.
            x: Integer[TT, " batch seq_len"] Current predicted sequence.
        """
        x: Integer[TT, " batch seq_len"] = results["x"]
        out = self.tokenizer.batch_decode(x, skip_special_tokens=True)
        out_with_spl_tokens: List[str] = self.tokenizer.batch_decode(
            x, skip_special_tokens=False
        )
        return out, out_with_spl_tokens, x

    def stop(
        self,
        step_results: MLMStepResults,
        current_step: int,
    ) -> bool:
        time_ended = not bool((step_results["t"] > 0).any())
        max_steps_reached = current_step >= self.max_steps
        return time_ended or max_steps_reached

    def predict_single_step(
        self,
        step_results: MLMStepResults,
        final_step: bool = False,
    ) -> MLMStepResults:
        """
        Args:
            step_results:
                x: Integer[TT, " batch seq_len"] Current predicted sequence.
                attention_mask: Bool[TT, " batch seq_len"] Mask of the current sequence.
                logits: Float[TT, " batch seq_len vocab_size"] Logits of the current sequence.
                t: Integer[TT, " batch"] Current timestep.
                constraint: Bool[TT, " batch seq_len"] Constraint of the current sequence.
                change: Bool[TT, " batch"] Whether any token in the current sequence is changed.
        Returns:
            MLMStepResults: Updated step results.
        """
        # fmt: off
        x_t: Integer[TT, " batch seq_len"] = step_results["x"]
        attention_mask: Bool[TT, " batch seq_len"] = step_results["attention_mask"]
        positions = attention_mask.cumsum(dim=1) - 1
        positions *= attention_mask
        constraint: Optional[Bool[TT, " batch seq_len"]] = step_results["constraint"]
        p_x0 = step_results["p_x0"]
        conf: Float[TT, " batch seq_len"] = step_results["conf"]
        # fmt: on
        s = t - self.dt
        # TODO (efficiency): Logits can be cached if the model does not depend on dot_sigma_t
        assert self.model is not None, "Model is not initialized"
        logits = self.model(x_t, attention_mask, positions)
        if p_x0 is None:
            p_x0 = logits.exp()
        if conf is None:
            conf = -torch.ones_like(x_t, dtype=p_x0.dtype) * torch.inf

        # predicting real tokens
        # TODO (compile): This if is not compile friendly. Split into two functions.
        if not final_step:
            chance_t = t[:, None, None]
            chance_s = s[:, None, None]
            alpha_t = (1 - chance_t)[0].item()
            alpha_s = (1 - chance_s)[0].item()

            if alpha_t > 0:
                sigma_max = min(1, (1 - alpha_s) / alpha_t)
            else:
                sigma_max = 1
            eta = conf.softmax(dim=-1)
            masked_flag = (x_t == self.tokenizer.mask_token_id).to(torch.bool)
            eta[masked_flag] = 0
            sigma = eta * sigma_max
            q_xs = p_x0 * (1 - sigma[:, :, None])
            q_xs[..., self.tokenizer.mask_token_id] = sigma
            q_xs_2 = p_x0 * (
                (alpha_s - (1 - sigma[:, :, None]) * alpha_t) / (1 - alpha_t)
            )
            q_xs_2[..., self.tokenizer.mask_token_id] = (
                1 - alpha_s - sigma * alpha_t
            ) / (1 - alpha_t)
            copy_flag = (x_t != self.tokenizer.mask_token_id).to(torch.bool)
            q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
            xs = self.sampling_function(q_xs)  # (*batch, seq_len)
            unmask_mask = (x_t == self.tokenizer.mask_token_id) & (
                xs != self.tokenizer.mask_token_id
            )
            batch_indices = torch.arange(xs.shape[0])[:, None]
            feature_indices = torch.arange(xs.shape[1])
            conf_values = -p_x0[batch_indices, feature_indices, xs]
            conf[unmask_mask] = conf_values[unmask_mask]
            remask_mask = (x_t != self.tokenizer.mask_token_id) & (
                xs == self.tokenizer.mask_token_id
            )
            conf[remask_mask] = -torch.inf
        else:
            q_xs = torch.softmax(
                logits, dim=-1
            )  # (*batch, seq_len, vocab_size)
            xs = torch.argmax(q_xs, dim=-1)  # (*batch, seq_len)

        # copy the input for input positions that were non-mask
        xs = torch.where(
            x_t == self.tokenizer.mask_token_id,
            xs,
            x_t,
        )
        return {
            "x": xs,
            "attention_mask": attention_mask,
            "t": s,
            "logits": logits,
            "constraint": constraint,
            "change": None,
            "conf": conf,
            "p_x0": p_x0,
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
        for text, text_with_spl_tokens, ids in zip(
            preds["text"],
            preds["text_with_spl_tokens"],
            preds["ids"].tolist(),
        ):
            dicts.append(
                {
                    "text": text,
                    "text_with_spl_tokens": text_with_spl_tokens,
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
        x = batch["input_ids"]
        _max_new_tokens = (x == self.tokenizer.mask_token_id).sum(dim=-1)
        if (_max_new_tokens[0] != _max_new_tokens).any():
            raise ValueError(
                "All sequences must have the same number of mask tokens"
            )
        self.max_new_tokens = int(_max_new_tokens[0])
        positions = batch["attention_mask"].cumsum(dim=1) - 1
        step_results: MLMStepResults = {
            "x": x.clone(),
            "attention_mask": batch["attention_mask"],
            "positions": positions,
            "logits": None,  # type: ignore ok for first step
            "done": torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
            "steps_taken": torch.zeros(
                x.shape[0], dtype=torch.int, device=x.device
            ),
            "change": torch.ones(
                x.shape[0], dtype=torch.bool, device=x.device
            ),
            "constraint": None,
        }
        current_step = 1
        while not self.stop(step_results, current_step):
            step_results = self.predict_single_step(
                step_results,
            )
            current_step += 1
        step_results = self.predict_single_step(
            step_results,
            final_step=True,
        )
        # decode the final step
        (
            out,
            out_with_spl_tokens,
            final_x,
        ) = self.decode(step_results)

        _end_time = time.time()
        _time_taken = _end_time - _start_time
        self.reset()
        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": final_x,
            "attention_mask": step_results["attention_mask"],
            "loss": None,
            "time_taken": [_time_taken]
            * len(out),  # cannot separate time for each sample
        }
