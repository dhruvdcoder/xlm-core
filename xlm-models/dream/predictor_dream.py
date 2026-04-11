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
from mlm.types_mlm import (
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
from mlm.unbatch import unbatch
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


class Allow:
    def __init__(self):
        self._allowed = None

    def reset(self):
        self._allowed = None

    def __call__(
        self,
        current_step: int,
        input_end_indices: Integer[TT, " batch"],
        seq_len: int,
    ) -> Bool[TT, "batch seq_len"]:
        if current_step == 0:
            if not (
                input_end_indices[0].unsqueeze(0) == input_end_indices
            ).all():
                raise ValueError(
                    "Currently only support starting prediction at the same index for the entire batch."
                )
            self._allowed = torch.arange(
                seq_len, device=input_end_indices.device
            ).unsqueeze(0) >= input_end_indices.unsqueeze(-1)
        assert self._allowed is not None
        return self._allowed


class BlockAllowSchedule(Allow):
    block_size: int
    max_new_tokens: int
    diffusion_steps: int
    _allowed: Bool[TT, "batch seq_len"]
    _current_block: int

    def __init__(
        self, block_size: int, max_new_tokens: int, diffusion_steps: int
    ):
        self.block_size = block_size
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.num_blocks, reminder = divmod(max_new_tokens, block_size)
        if reminder:
            raise ValueError(
                f"{max_new_tokens=} must be a multiple of {block_size=}"
            )

        self.steps_per_block, reminder = divmod(
            diffusion_steps, self.num_blocks
        )
        if reminder:
            raise ValueError(
                f"{diffusion_steps=} must be a multiple of {self.num_blocks=}. We currently only support this."
            )
        self.tokens_per_step, reminder = divmod(
            max_new_tokens, diffusion_steps
        )
        if reminder:
            raise ValueError(
                f"{max_new_tokens=} must be a multiple of {diffusion_steps=}"
            )
        # ensure a block is fully done before moving on
        assert block_size % self.tokens_per_step == 0

        self._current_block = 0
        self._input_end_index: Optional[int] = None
        self._allowed = None

    def reset(self):
        self._current_block = 0
        self._input_end_index = None
        self._allowed = None

    def __call__(
        self,
        current_step: int,
        input_end_indices: Integer[TT, " batch"],
        seq_len: int,
    ) -> Bool[TT, "batch seq_len"]:

        if current_step == 0:
            if not (
                input_end_indices[0].unsqueeze(0) == input_end_indices
            ).all():
                raise ValueError(
                    "Currently only support starting prediction at the same index for the entire batch."
                )
            self._input_end_index = int(
                input_end_indices[0].item()
            )  # not compile-friendly
            # initialize the disallow changing prompt
            self._current_block = 0
            self._allowed = torch.zeros(
                (input_end_indices.shape[0], seq_len),
                device=input_end_indices.device,
                dtype=torch.bool,
            )
            i = self._input_end_index
            j = self._input_end_index + self.block_size
            self._allowed[:, i:j] = 1
        else:
            current_block, reminder = divmod(
                current_step * self.tokens_per_step, self.block_size
            )
            assert self._input_end_index is not None
            if current_block != self._current_block:
                # disallow changing the block we just finished
                i = (
                    self._input_end_index
                    + self._current_block * self.block_size
                )
                j = (
                    self._input_end_index
                    + (self._current_block + 1) * self.block_size
                )
                self._allowed[:, i:j] = 0
                # allow changing the new block we are entering
                i = self._input_end_index + current_block * self.block_size
                j = (
                    self._input_end_index
                    + (current_block + 1) * self.block_size
                )
                self._allowed[:, i:j] = 1

                self._current_block = current_block
        return self._allowed


def _create_generate_until_tensors(
    tokenizer: Tokenizer,
    generate_until: List[str],
    device: torch.device,
) -> Tuple[Integer[TT, " batch max_patch_len"], Bool[TT, " batch max_patch_len"]]:
    res = tokenizer(
        generate_until,
        add_special_tokens=False,
        padding="longest",
        padding_side="right",
        return_attention_mask=True,
    )
    return (
        torch.tensor(res["input_ids"], dtype=torch.long, device=device),
        ~torch.tensor(res["attention_mask"], dtype=torch.bool, device=device),
    )


def check_batched_generate_until(
    generate_until: Integer[TT, " n max_patch_len"],
    generate_until_mask: Bool[TT, " n max_patch_len"],
    ids: Integer[TT, " batch seq_len"],
) -> Bool[TT, " batch seq_len"]:
    """Check if any generate_until token pattern appears in ids.

    Args:
        generate_until: (n, max_patch_len) contiguous token-id chunks to look for.
        generate_until_mask: Pad mask (True = pad, False = valid) matching generate_until.
        ids: (batch, seq_len) generated token ids.
    """
    n, w = generate_until.shape
    b, seq_len = ids.shape
    unfolded = ids.unfold(-1, w, 1)  # (b, seq-w+1, w)
    match_ = unfolded.unsqueeze(-2) == generate_until[None, None, :, :]  # (b, seq-w+1, n, w)
    match_ = match_.logical_or(generate_until_mask[None, None, :, :])
    temp = match_.all(-1)  # (b, seq-w+1, n)
    matched = temp.any(-1)  # (b, seq-w+1)
    mask = torch.cat(
        [matched, torch.zeros(b, w - 1, dtype=matched.dtype, device=matched.device)],
        dim=-1,
    )  # (b, seq)
    return mask


class DreamPredictor(torch.nn.Module, Predictor[MLMBatch, MLMPredictionDict]):
    """Same as MLMPredictor for now."""

    def __init__(
        self,
        max_steps: int,
        max_new_tokens: Optional[int] = None,
        block_size: Optional[int] = 8,
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
        generate_until: Optional[List[str]] = None,
        force_fill_max: Optional[int] = 3,
    ):
        """Initialize Dream Predictor.

        Args:
            max_steps: Maximum number of prediction steps (diffusion steps).
            max_new_tokens: Number of mask tokens to append to the prompt.
            block_size: If set, use BlockAllowSchedule to unmask one block at a
                time instead of all positions simultaneously. Must divide
                max_new_tokens evenly.
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
            generate_until: Optional list of token strings (e.g. EOS); stop early when any appears in the output with no masks before it.
            force_fill_max: After generate_until is hit this many times, force stop even if masks remain before it.
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
        self.confidence_temperature = confidence_temperature
        self.temperature = temperature
        self.logits_hook = logits_hook
        self.flash_attn = None

        if block_size is not None:
            assert (
                max_new_tokens is not None
            ), "max_new_tokens is required when block_size is set"
            self.allowed: Allow = BlockAllowSchedule(
                block_size=block_size,
                max_new_tokens=max_new_tokens,
                diffusion_steps=max_steps,
            )
        else:
            self.allowed = Allow()

        self._generate_until_strings = generate_until
        self._generate_until: Optional[Integer[TT, " n max_patch_len"]] = None
        self._generate_until_mask: Optional[Bool[TT, " n max_patch_len"]] = None
        self.force_fill_max = force_fill_max
        self.force_fill_counter: Optional[Integer[TT, " batch"]] = None

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

    def _ensure_generate_until_tensors(self, device: torch.device) -> None:
        """Lazily build generate_until tensors on first use (needs tokenizer + device)."""
        if self._generate_until is not None:
            return
        if self._generate_until_strings is None:
            return
        tokenizer = self._require_tokenizer()
        self._generate_until, self._generate_until_mask = (
            _create_generate_until_tensors(
                tokenizer, list(self._generate_until_strings), device
            )
        )

    def first_generate_until_hit(
        self, step_results: MLMStepResults
    ) -> Tuple[Bool[TT, " batch seq_len"], Integer[TT, " batch"]]:
        assert self._generate_until is not None and self._generate_until_mask is not None
        x = step_results["x"]
        device = x.device
        self._generate_until = self._generate_until.to(device=device)
        self._generate_until_mask = self._generate_until_mask.to(device=device)
        batch_size, seq_len = x.shape
        isin = check_batched_generate_until(
            self._generate_until, self._generate_until_mask, x
        )
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        isin = isin.logical_and(
            arange > step_results["input_end_positions"].unsqueeze(-1)
        )
        pos = torch.where(isin, arange, seq_len)
        first_pos = pos.min(dim=-1).values
        return isin, first_pos

    def check_generate_until(
        self, step_results: MLMStepResults
    ) -> Bool[TT, " batch"]:
        if self._generate_until is None:
            batch_size = step_results["x"].shape[0]
            return torch.zeros(batch_size, dtype=torch.bool, device=step_results["x"].device)

        x = step_results["x"]
        device = x.device
        batch_size, seq_len = x.shape
        tokenizer = self._require_tokenizer()

        if self.force_fill_counter is None:
            self.force_fill_counter = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )

        isin, first_pos = self.first_generate_until_hit(step_results)
        hit = isin.any(-1).long()
        self.force_fill_counter += hit

        mask = x == tokenizer.mask_token_id
        mask_cumsum = mask.cumsum(dim=-1)
        pos_minus1 = (first_pos - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=device)
        masked_before = mask_cumsum[batch_idx, pos_minus1]

        return ((first_pos < seq_len) & (masked_before == 0)).logical_or(
            self.force_fill_counter >= self.force_fill_max
        )

    def reset(self):
        self.allowed.reset()
        self.force_fill_counter = None

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
    ) -> bool:
        if step_results["current_step"] >= self.max_steps - 1:
            return True
        tokenizer = self._require_tokenizer()
        should_stop = (step_results["x"] != tokenizer.mask_token_id).all(dim=-1)
        if self._generate_until is not None:
            stop_using_gen_until = self.check_generate_until(step_results)
            should_stop = stop_using_gen_until.logical_or(should_stop)
        step_results["done"].logical_or_(should_stop)
        return step_results["done"].all().item()

    def predict_single_step(
        self,
        step_results: MLMStepResults,
        final_step: bool = False,
    ) -> MLMStepResults:
        # fmt: off
        attention_mask: Bool[TT, " batch seq_len"] = step_results["attention_mask"]
        x = step_results["x"]
        positions = step_results["positions"]
        current_step = step_results["current_step"]
        input_end_positions = step_results["input_end_positions"]
        # fmt: on
        tokenizer = self._require_tokenizer()
        assert self.model is not None, "Model is not initialized"
        logits = self.model(
            x, attention_mask if not self.flash_attn else None, positions
        )
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

        return {
            "x": x,
            "attention_mask": attention_mask,
            "positions": positions,
            "logits": logits,
            "current_step": current_step + 1,
            "input_end_positions": input_end_positions,
            "done": step_results.get("done", torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)),
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
        input_end_positions = torch.full(
            (x.shape[0],), output_start_idx, device=x.device, dtype=torch.long
        )

        self._ensure_generate_until_tensors(x.device)

        step_results: MLMStepResults = {
            "x": x,
            "attention_mask": attention_mask,
            "positions": positions,
            "logits": None,  # type: ignore # ok for first step
            "current_step": 0,
            "input_end_positions": input_end_positions,
            "done": torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
        }
        if self.flash_attn is None:
            self.flash_attn = getattr(self.model, "force_flash_attn", False)
        if attention_mask is not None:
            if ~attention_mask.any() and self.flash_attn:
                raise ValueError(
                    "Attention mask is not all True and flash attention is enabled"
                )

        batch_size = x.shape[0]
        step_count = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        time_taken = torch.zeros(batch_size, dtype=torch.float64, device=x.device)

        while not self.stop(step_results):
            step_results = self.predict_single_step(step_results)
            not_done = ~step_results["done"]
            step_count += not_done.to(dtype=torch.long)
            time_taken[not_done] = time.time() - _start_time
            if flags.DEBUG_PRINT_PREDS:
                print(
                    f"Step {step_results['current_step']}: {remove_trailing_pads(tokenizer.decode(step_results['x'][0], skip_special_tokens=False), tokenizer, tokens_to_remove=[tokenizer.mask_token])}",
                    flush=True,
                )
        step_results = self.predict_single_step(
            step_results,
            final_step=True,
        )
        step_count += 1  # final step counts for everyone
        final_time = time.time() - _start_time
        time_taken[time_taken == 0] = final_time

        (out, final_x) = self.decode(step_results)

        self.reset()
        if flags.DEBUG_PRINT_PREDS:
            print(
                f"Final Step {step_results['current_step']}: {remove_trailing_pads(tokenizer.decode(final_x[0], skip_special_tokens=False), tokenizer)}",
                flush=True,
            )
            print("-" * 100, flush=True)
        return {
            "text": out,
            "ids": final_x,
            "loss": None,
            "time_taken": time_taken.tolist(),
            "output_start_idx": output_start_idx,
            "steps_taken": step_count.tolist(),
        }
