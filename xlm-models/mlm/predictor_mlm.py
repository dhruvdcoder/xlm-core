from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)
from functools import partial
import torch
from jaxtyping import Bool, Integer, Float
from xlm import flags
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
from .unbatch import unbatch
from xlm.utils.text import remove_trailing_pads

MLMStepResults = Dict[str, Any]


class LogitsShiftBy1:
    """Map next-token logits to per-position logits (Dream-style alignment).

    ``output[..., i]`` is taken from the backbone's prediction for the following
    position so it lines up with token ``i``. Use as ``logits_hook`` in Hydra:

    ``logits_hook: { _target_: mlm.predictor_mlm.LogitsShiftBy1 }``
    """

    def __call__(self, logits: TT) -> TT:
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


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
        confidence: Optional[
            Literal["prob_diff", "entropy", "top_prob"]
        ] = None,
        threshold: Optional[float] = None,
        confidence_temperature: float = 1.0,
        temperature: float = 1.0,
        logits_hook: Optional[Callable[[TT], TT]] = None,
        skip_special_tokens: bool = True,
    ):
        """Initialize MLM Predictor.

        Args:
            max_steps: Maximum number of prediction steps.
            tokenizer: Tokenizer for encoding/decoding; if omitted, set by Harness after instantiate.
            noise_schedule: Noise schedule for the diffusion process.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            confidence: Confidence-based position sampling parameter.
            threshold: Threshold for confidence-based position sampling.
            confidence_temperature: Temperature for stochastic position selection; lower values concentrate on most confident positions.
            temperature: Token sampling temperature; 0 selects greedy top-1 (via top_k=1).
            logits_hook: Optional transform applied to model logits before masking/sampling.
            model: The MLM model to use for predictions.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        if temperature == 0:
            self.sampling_function = partial(sample_from_top_k, 1)
        elif top_k is None and top_p is None:
            self.sampling_function = partial(
                sample_from_logits, temperature=temperature
            )
        elif top_k is not None and top_p is None:
            self.sampling_function = partial(
                sample_from_top_k, top_k, temperature=temperature
            )
        elif top_k is None and top_p is not None:
            self.sampling_function = partial(
                sample_from_top_p, top_p, temperature=temperature
            )
        else:
            raise ValueError("Both top_k and top_p cannot be non-None")

        self.noise_schedule = noise_schedule
        self.skip_special_tokens = skip_special_tokens
        self.confidence = confidence
        self.threshold = threshold
        if confidence is not None and threshold is None:
            raise ValueError(
                "threshold must be provided when confidence-based position "
                "selection is enabled (confidence is not None)."
            )
        self.confidence_temperature = confidence_temperature
        self.temperature = temperature
        self.logits_hook = logits_hook
        self.flash_attn = None

    def _require_tokenizer(self) -> Tokenizer:
        if self.tokenizer is None:
            raise ValueError("tokenizer is required")
        return self.tokenizer

    def _compute_confidence(
        self,
        logits: TT,
        masked: Bool[TT, " batch seq_len"],
    ) -> TT:
        """Per-position confidence score (higher = more confident). -inf at non-mask."""
        probs = logits.softmax(dim=-1)
        if self.confidence == "top_prob":
            score = probs.max(dim=-1)[0]
        elif self.confidence == "prob_diff":
            top2, _ = torch.topk(probs, k=2, dim=-1)
            score = top2[..., 0] - top2[..., 1]
        elif self.confidence == "entropy":
            score = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        else:
            raise ValueError(f"Unknown confidence: {self.confidence}")
        score = score.clone()
        score[~masked] = float("-inf")
        return score

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
        tokenizer = self._require_tokenizer()
        x: Integer[TT, " batch seq_len"] = results["x"]
        out_with_spl_tokens: List[str] = tokenizer.batch_decode(
            x, skip_special_tokens=self.skip_special_tokens
        )
        return out_with_spl_tokens, x

    def stop(
        self,
        step_results: MLMStepResults,
    ) -> Bool[TT, " batch"]:
        tokenizer = self._require_tokenizer()
        steps_done = bool(step_results["done"].all())
        all_filled = bool((step_results["x"] != tokenizer.mask_token_id).all())
        return steps_done or all_filled

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
        tokenizer = self._require_tokenizer()
        assert self.model is not None, "Model is not initialized"
        logits = self.model(
            x, attention_mask if not self.flash_attn else None, positions
        )
        if self.logits_hook is not None:
            logits = self.logits_hook(logits)
        masked = x == tokenizer.mask_token_id
        steps_left = self.max_steps - step_results["steps_taken"]
        if not final_step:
            if self.confidence is not None and self.threshold is not None:
                # Dynamic num_unmask via cumulative threshold (legacy MLM behavior)
                if self.confidence == "prob_diff":
                    temp = logits.softmax(dim=-1)  # (B, L, V)
                    top2_probs, _ = torch.topk(temp, k=2, dim=-1)  # (B, L, 2)
                    confidence = (
                        top2_probs[:, :, 0] - top2_probs[:, :, 1]
                    )  # (B, L)
                    confidence = (
                        1 - confidence
                    )  # prepare to sort in ascending order
                    confidence[~masked] = 1  # ignore non-masked positions
                    sorted_values, sorted_indices = torch.sort(confidence)
                    unmask = (
                        torch.cumsum(sorted_values, dim=-1)
                        - torch.cummax(sorted_indices, dim=-1).values
                        < self.threshold
                    )
                    unmask = unmask.scatter(-1, sorted_indices, unmask)
                elif self.confidence == "entropy":
                    temp = logits.softmax(dim=-1)  # (B, L, V)
                    confidence = torch.sum(
                        temp * torch.log(temp + 1e-10), dim=-1
                    )  # (B, L)
                    raise NotImplementedError(
                        "Entropy-based sampling is not implemented"
                    )
                elif self.confidence == "top_prob":
                    confidence = logits.softmax(dim=-1).max(dim=-1)[
                        0
                    ]  # (B, L)
                    confidence = (
                        1 - confidence
                    )  # prepare to sort in ascending order
                    confidence[~masked] = 1  # ignore non-masked positions
                    sorted_values, sorted_indices = torch.sort(
                        confidence, dim=-1, descending=False
                    )
                    unmask = (
                        torch.cumsum(sorted_values, dim=-1)
                        - torch.cummax(sorted_values, dim=-1).values
                        < self.threshold
                    )
                    unmask = unmask.scatter(-1, sorted_indices, unmask)
                else:
                    raise ValueError(f"Unknown confidence: {self.confidence}")
            elif self.confidence is not None:
                num_unmask = (
                    masked.sum(dim=-1) / (steps_left).clamp(min=1)
                ).long()
                score = self._compute_confidence(logits, masked)
                selection_mode = (
                    "greedy" if self.confidence_temperature == 0 else "sample"
                )
                unmask = select_random_indices(
                    inp_shape=x.shape,
                    num_unmask=num_unmask,
                    select_from_mask=masked,
                    selection_score=score,
                    selection_mode=selection_mode,
                    temperature=self.confidence_temperature,
                )
            else:
                num_unmask = (
                    masked.sum(dim=-1) / (steps_left).clamp(min=1)
                ).long()
                unmask = select_random_indices(
                    inp_shape=x.shape,
                    num_unmask=num_unmask,
                    select_from_mask=masked,
                    selection_score=None,  # uniform
                    selection_mode="sample",
                    temperature=self.confidence_temperature,
                )
        else:
            unmask = masked
        # can unmask only masked positions
        unmask = masked.logical_and(unmask)
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
        dicts: List[Dict[str, Any]] = unbatch(preds, length=len(preds["text"]))
        return dicts

    @torch._dynamo.disable()
    def predict(
        self,
        batch: MLMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMPredictionDict:
        tokenizer = self._require_tokenizer()
        _start_time = time.time()
        x = batch["input_ids"].clone()
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                x.shape[0], x.shape[1], dtype=torch.bool, device=x.device
            )
        else:
            attention_mask = attention_mask.to(
                device=x.device, dtype=torch.bool
            )

        output_start_idx = x.shape[-1]
        if self.max_new_tokens is not None:
            # we assume prefix is left-padded
            x = torch.cat(
                [
                    x,
                    torch.full(
                        (x.shape[0], self.max_new_tokens),
                        tokenizer.mask_token_id,
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
                        dtype=torch.bool,
                    ),
                ],
                dim=-1,
            )
        positions = (
            (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0).long()
        )
        steps_taken = torch.zeros(x.shape[0], dtype=torch.int, device=x.device)

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
        if self.flash_attn is None:
            self.flash_attn = getattr(self.model, "force_flash_attn", False)
        # check if there are any False in the attention mask
        if attention_mask is not None:
            if ~attention_mask.any() and self.flash_attn:
                raise ValueError(
                    "Attention mask is not all True and flash attention is enabled"
                )

        while not self.stop(step_results):
            has_masked = (step_results["x"] == tokenizer.mask_token_id).any(
                dim=-1
            )
            steps_taken[has_masked] += 1
            step_results = self.predict_single_step(
                step_results,
            )
            if flags.DEBUG_PRINT_PREDS:
                print(
                    f"Step {steps_taken[0].item()}: {remove_trailing_pads(tokenizer.decode(step_results['x'][0], skip_special_tokens=False), tokenizer, tokens_to_remove=[tokenizer.mask_token])}",
                    flush=True,
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
        if flags.DEBUG_PRINT_PREDS:
            print(
                f"Final Step {steps_taken[0].item()}: {remove_trailing_pads(tokenizer.decode(final_x[0], skip_special_tokens=False), tokenizer)}",
                flush=True,
            )
            print("-" * 100, flush=True)
        return {
            "text": out,
            "ids": final_x,
            "loss": None,
            "time_taken": [_time_taken]
            * len(out),  # cannot separate time for each sample
            "output_start_idx": output_start_idx,
            "steps_taken": steps_taken.tolist(),
        }
