from typing import (
    Any,
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
from xlm.datamodule import Tokenizer
from torch import Tensor as TT
from .types_flexmdm import (
    FlexMDMBatch,
    FlexMDMPredictionDict,
    FlexMDMModel,
    FlexMDMSeq2SeqPredictionBatch,
)
from xlm import flags
from xlm.harness import Predictor, PredictorHistoryMixin
from xlm.noise import NoiseSchedule
from xlm.log_predictions import TokensHook
from xlm.utils.nn import (
    sample_from_top_k,
    sample_from_top_p,
    sample_from_logits,
    sample_categorical,
)
import time

FlexMDMStepResults = Dict[str, Any]


class FlexMDMTokensHook(TokensHook):
    def __call__(
        self,
        metadata: Dict[str, Any],
        x_inp: Float[TT, " batch seq_len"],
        x_out: Float[TT, " batch seq_len"],
    ) -> Float[TT, " batch seq_len"]:
        return x_out


class FlexMDMPredictor(
    torch.nn.Module,
    PredictorHistoryMixin,
    Predictor[FlexMDMBatch, FlexMDMPredictionDict],
):
    """Base predictor for FlexMDM. Stochastically selects positions to insert and unmask based on max_steps and max_new_tokens."""

    tokens_hook: Optional[TokensHook] = None

    def __init__(
        self,
        max_steps: int,
        max_new_tokens: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[FlexMDMModel] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        suppress_pad_token: Optional[int] = None,
        len_predict_type: str = "distribution",
        confidence: Literal[
            "position", "top_prob", "prob_diff", "entropy", None
        ] = None,
        return_history: bool = False,
    ):
        """Initialize FlexMDM Predictor.

        Args:
            max_steps: Maximum number of prediction steps.
            max_new_tokens: Maximum number of new tokens to generate.
            tokenizer: Tokenizer for encoding/decoding.
            model: The FlexMDM model to use for predictions.
            noise_schedule: Noise schedule for the diffusion process.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            suppress_pad_token: Number of steps for which to suppress the pad token.
            len_predict_type: Type of length prediction ("distribution" or "expectation").
            confidence: Confidence-based decoding method (None for tau-leaping).
            return_history: Whether to track and return generation history.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required")

        super().__init__()
        self.init_history(return_history=return_history, decode_fn=self.decode)
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.dt = (1 - 1e-5) / (max_steps + 1)
        self.top_k = top_k
        self.top_p = top_p
        if top_k is None and top_p is None:
            self.sampling_function = sample_from_logits
        elif top_k is not None and top_p is None:
            self.sampling_function = partial(sample_from_top_k, top_k)
        elif top_k is None and top_p is not None:
            self.sampling_function = partial(sample_from_top_p, top_p)
        else:
            raise ValueError("Both top_k and top_p cannot be non-None")

        self.noise_schedule = noise_schedule
        self.suppress_pad_token = suppress_pad_token
        self.len_predict_type = len_predict_type
        self.confidence = confidence
        self.tokens_hook = None

    def reset(self):
        # simple predictor has no state
        pass

    def decode(self, results: FlexMDMStepResults) -> Tuple[
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
        step_results: FlexMDMStepResults,
    ) -> Bool[TT, " batch"]:
        time_ended = not bool((step_results["t"] > 0).any())
        all_filled = not (
            (step_results["x_t"] == self.tokenizer.mask_token_id).any()
        )
        return step_results["done"].all().item() or time_ended or all_filled

    def to_dict(
        self,
        batch: FlexMDMBatch,
        preds: FlexMDMPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convert predictions to a list of dictionaries."""
        from itertools import cycle

        # Format history using the mixin helper
        history_data = preds.get(
            "history", [[] for _ in range(len(preds["text"]))]
        )
        formatted_history = self.format_history_for_output(
            history_data, round_precision=4
        )
        to_zip = [
            preds["text"],
            preds["ids"].tolist(),
            formatted_history,
            preds.get("time_taken", cycle([-1])),
        ]
        metric_keys = []
        for n in preds:
            if n.startswith("metric_"):
                metric_keys.append(n)
                to_zip.append(preds[n])

        preds_list: List[Tuple[str, List[int], List[List[Any]], float]] = list(
            zip(*to_zip)
        )
        dicts: List[Dict[str, Any]] = []
        for preds_ in preds_list:
            dicts.append(
                {
                    "text": preds_[0],
                    "ids": preds_[1],
                    "history": preds_[2],
                    "time_taken": preds_[3],
                    **{k: preds_[4 + i] for i, k in enumerate(metric_keys)},
                }
            )
        return dicts

    @torch._dynamo.disable()
    def predict(
        self,
        batch: FlexMDMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FlexMDMPredictionDict:
        _start_time = time.time()
        if flags.DEBUG_PRINT_PREDS:
            with open("temp.txt", "a") as f:
                f.write("-" * 100 + "\n")

        xt = batch["input_ids"].clone()
        fixed_gaps = batch[
            "fixed"
        ].clone()  # 1's at gaps to the left of prefix tokens which can't be expanded, 0's everywhere else
        attention_mask = (
            (xt != self.tokenizer.pad_token_id).bool().to(xt.device)
        )

        batch_size, max_length = xt.shape
        device = xt.device
        mask, pad = self.tokenizer.mask_token_id, self.tokenizer.pad_token_id
        steps = self.max_steps
        t = torch.zeros(batch_size, device=device)
        dt = self.dt

        # Initialize history tracking
        history: List[List[Tuple[str, float, int]]] = self.create_history(
            batch_size
        )
        # Record initial state (step 0)
        history = self.update_history_explicit(
            history,
            self.tokenizer.batch_decode(xt, skip_special_tokens=True),
            t.tolist(),
            0,
        )

        # Precompute row indices for scatter
        batch_idx_L = (
            torch.arange(batch_size, device=device)
            .view(batch_size, 1)
            .expand(batch_size, max_length)
        )
        pos_idx_L = (
            torch.arange(max_length, device=device)
            .view(1, max_length)
            .expand(batch_size, max_length)
        )

        for i in range(steps):
            # # --- predict rates ---
            if self.tokens_hook is not None:
                xt_inp = xt
            else:
                xt_inp = None
            unmask_rate_, length_logits = self.model(
                xt,
                t,
                attention_mask,
            )  # (B, L, V), (B, L)
            unmask_rate = (
                unmask_rate_.softmax(-1)
                * self.noise_schedule.unmasking_noise_schedule.rate_scale_factor(
                    t
                )[
                    :, None, None
                ]
            )
            if self.len_predict_type == "distribution":
                len_rate = (
                    length_logits.softmax(-1)
                    * torch.arange(
                        0, max_length, device=length_logits.device
                    ).view(1, 1, -1)
                ).sum(
                    -1
                ) * self.noise_schedule.insertion_noise_schedule.rate_scale_factor(
                    t
                ).unsqueeze(
                    1
                )
            elif self.len_predict_type == "expectation":
                # length_logits is already the expected length of shape (B, L)
                len_rate = length_logits * self.noise_schedule.insertion_noise_schedule.rate_scale_factor(
                    t
                ).unsqueeze(
                    1
                )

            if i == steps - 1:
                # last step: deterministic unmask via argmax
                mask_pos = xt == mask
                new_token = unmask_rate.argmax(dim=2)
                new_xt = xt.clone()
                new_xt[mask_pos] = new_token[mask_pos]
                new_xt = torch.where(xt == pad, pad, new_xt)
                new_xt = torch.where((xt != mask) & (xt != pad), xt, new_xt)
                xt = new_xt
                t = t + dt
                if flags.DEBUG_PRINT_PREDS:
                    with open("temp.txt", "a") as f:
                        f.write(f"t: {t}\n")
                        _decoded = self.tokenizer.batch_decode(
                            xt, skip_special_tokens=False
                        )
                        for seq in _decoded:
                            f.write(f"x[0]: {seq}\n")
                        f.write("\n")
                continue

            # --- confidence-based decoding ---
            if self.confidence is not None:
                # Confidence-based unmasking (vectorized)
                mask_positions = xt == mask  # (B, L)
                num_mask_positions = mask_positions.sum(dim=1)  # (B,)

                # 1. Determine number of tokens to unmask using Poisson
                unmask_counts = torch.poisson(
                    num_mask_positions.float() * dt
                ).long()  # (B,)

                # 2. Calculate confidence based on selected method
                if self.confidence == "position":
                    # Position-based confidence: position i / len(xt)
                    xt_len = (xt != pad).sum(
                        dim=1
                    )  # (B,) - current sequence lengths
                    position_indices = (
                        torch.arange(max_length, device=device)
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                    )  # (B, L)
                    confidence = 1.0 - (
                        position_indices.float()
                        / xt_len.unsqueeze(1).float().clamp(min=1)
                    )  # (B, L)

                elif self.confidence == "top_prob":
                    # Top probability confidence
                    import torch.nn.functional as F

                    token_logits = unmask_rate  # (B, L, V) - use the unmask_rate as logits
                    unmask_probs = F.softmax(token_logits, dim=-1)  # (B, L, V)
                    confidence = unmask_probs.max(dim=-1)[0]  # (B, L)

                elif self.confidence == "prob_diff":
                    # Probability difference confidence (top - second top)
                    import torch.nn.functional as F

                    token_logits = unmask_rate  # (B, L, V)
                    unmask_probs = F.softmax(token_logits, dim=-1)  # (B, L, V)
                    top2_probs, _ = torch.topk(
                        unmask_probs, k=2, dim=-1
                    )  # (B, L, 2)
                    confidence = (
                        top2_probs[:, :, 0] - top2_probs[:, :, 1]
                    )  # (B, L)

                elif self.confidence == "entropy":
                    # Entropy-based confidence (lower entropy = higher confidence)
                    import torch.nn.functional as F

                    token_logits = unmask_rate  # (B, L, V)
                    unmask_probs = F.softmax(token_logits, dim=-1)  # (B, L, V)
                    entropy = -torch.sum(
                        unmask_probs * torch.log(unmask_probs + 1e-10), dim=-1
                    )  # (B, L)
                    confidence = (
                        -entropy
                    )  # (B, L) - negative entropy so lower entropy gives higher confidence

                else:
                    raise ValueError(f"Unknown confidence: {self.confidence}")

                # No window constraint - only mask positions are eligible
                confidence = torch.where(
                    mask_positions,
                    confidence,
                    torch.tensor(-float("inf"), device=device),
                )

                new_xt = xt.clone()

                # Vectorized unmasking
                max_unmask = unmask_counts.max().item()
                if max_unmask > 0:
                    # Get top-k indices for all batches
                    _, all_top_indices = torch.topk(
                        confidence, k=max_unmask, dim=1, largest=True
                    )  # (B, max_unmask)

                    # Create mask for valid unmask operations
                    unmask_mask = torch.arange(
                        max_unmask, device=device
                    ).unsqueeze(0) < unmask_counts.unsqueeze(
                        1
                    )  # (B, max_unmask)

                    # Get most likely tokens
                    # most_likely_tokens = unmask_rate.argmax(dim=-1)  # (B, L)
                    most_likely_tokens = self.sampling_function(
                        unmask_rate_
                    )  # (B, L)

                    # Gather the tokens to place at selected positions
                    selected_positions = all_top_indices[
                        unmask_mask
                    ]  # Flattened valid positions
                    batch_indices = (
                        torch.arange(batch_size, device=device)
                        .unsqueeze(1)
                        .expand(-1, max_unmask)[unmask_mask]
                    )  # Corresponding batch indices

                    # Apply unmasking with sampled tokens
                    new_xt[batch_indices, selected_positions] = (
                        most_likely_tokens[batch_indices, selected_positions]
                    )

            else:
                # --- tau-leaping unmask via Poisson ---
                counts = torch.poisson(unmask_rate * dt).long()
                mask_pos = xt == mask
                counts[~mask_pos.unsqueeze(-1).expand_as(counts)] = 0
                counts[..., mask] = 0
                sum_c = counts.sum(dim=2)
                one_event = sum_c == 1
                if self.top_k is None and self.top_p is None:
                    new_token = counts.argmax(dim=2)
                else:
                    new_token = self.sampling_function(unmask_rate_)
                new_xt = xt.clone()
                new_xt[one_event] = new_token[one_event]
                new_xt = torch.where(xt == pad, pad, new_xt)
                new_xt = torch.where((xt != mask) & (xt != pad), xt, new_xt)

            # insertion only on non-last
            if i != steps - 1:
                # --- Poisson insertion, compute new lengths and fill masks ---
                ext = torch.poisson(len_rate * dt).long()  # (B, L)
                xt_len = xt.ne(pad).sum(dim=1)  # (B,)
                gaps = torch.arange(max_length, device=device).view(1, -1)
                ext = (
                    (1 - fixed_gaps)
                    * ext
                    * (gaps < xt_len.view(batch_size, 1)).long()
                )  # zero out insertions at fixed gaps (prefix) and beyond xt_len
                total_ext = ext.sum(dim=1)
                valid = xt_len + total_ext <= max_length
                ext = ext * valid.view(batch_size, 1).long()

                # compute prefix sums of insertions
                ext_ex = ext.int().cumsum(dim=1)  # (B, L)
                new_len = xt_len + total_ext  # (B,)

                # initialize with pads, then fill mask up to new_len
                xt_tmp = torch.full_like(xt, pad)
                mask_pos = pos_idx_L < new_len.view(batch_size, 1)
                xt_tmp[mask_pos] = mask

                # shift and scatter original tokens
                new_pos_orig = pos_idx_L + ext_ex[:, :max_length]  # (B, L)
                orig_mask = pos_idx_L < xt_len.view(batch_size, 1)
                flat_b = batch_idx_L[orig_mask]
                flat_p = new_pos_orig[orig_mask]
                xt_tmp[flat_b, flat_p] = new_xt[orig_mask]
            else:
                xt_tmp = new_xt

            xt = xt_tmp
            attention_mask = xt != self.tokenizer.pad_token_id
            t = t + dt
            if self.tokens_hook is not None:
                xt = self.tokens_hook(
                    metadata={"t": t, "step": i},
                    x_inp=xt_inp,  # type: ignore
                    x_out=xt,
                )

            # Update history after each step
            history = self.update_history_explicit(
                history,
                self.tokenizer.batch_decode(xt, skip_special_tokens=True),
                t.tolist(),
                i + 1,
            )

            if flags.DEBUG_PRINT_PREDS:
                with open("temp.txt", "a") as f:
                    f.write(f"t: {t}\n")
                    _decoded = self.tokenizer.batch_decode(
                        xt, skip_special_tokens=False
                    )
                    for _seq_idx, seq in enumerate(_decoded):
                        f.write(f"x[{_seq_idx}]: {seq}\n\n")
                    f.write("\n")

        out = self.tokenizer.batch_decode(xt, skip_special_tokens=True)
        # print("out")
        # print(out)

        _end_time = time.time()
        _time_taken = _end_time - _start_time
        self.reset()
        return {
            "text": out,
            "ids": xt,
            "loss": None,
            "time_taken": [_time_taken]
            * len(
                out
            ),  # cannot separate time for each sample, so run with batch_size=1 for time analysis
            "history": history,
            # "output_start_idx": output_start_idx,
        }

    @torch.inference_mode()
    def generate(self, prompts: List[str]) -> List[str]:
        batch = self.prepare_batch_for_generation(prompts)
        return self.predict(batch)["text"]
