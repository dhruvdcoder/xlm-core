"""Predictor for IDLM model.

This file implements the prediction logic for IDLM (Inserting Diffusion Language Model).
Simplified from PCDD implementation and adapted for XLM framework with token dropping.
"""

from typing import Any, Dict, List, Optional, Tuple, cast, Literal
from functools import partial
import torch
from jaxtyping import Bool, Integer, Float
from torch import Tensor as TT

from xlm.datamodule import Tokenizer
from xlm.harness import Predictor
from xlm.noise import NoiseSchedule
from .types import (
    IdlmBatch,
    IdlmPredictionDict,
    IdlmModel,
)
from xlm.lm.ilm.nn import (
    _remove_tokens,
    general_sample_over_last_two_dims,
)
from .collators import prepare_prefix_ids_idlm
from xlm.utils.nn import (
    sample_from_logits,
    sample_from_top_k,
    sample_from_top_p,
)
from .nn import hyp1f1_1_nplus1_vec


def incomplete_gamma_factor_using_series(
    n: Integer[TT, " batch"], dot_sigma: Float[TT, " batch"], K: int = 20
) -> Float[TT, " batch"]:
    """Compute incomplete gamma factor using series expansion."""
    return hyp1f1_1_nplus1_vec(dot_sigma, n, K=K)


class IdlmPredictorUtilitiesMixin:
    """Utility methods for IDLM predictor."""

    tokenizer: Tokenizer
    return_history: bool

    def clean_up_pred_ids(
        self,
        pred_ids: Integer[TT, " *batch seq_len"],
        hold_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
    ) -> Integer[TT, " *batch seq_len"]:
        """Remove mask tokens inserted due to batched prediction."""
        pad_token_id = self.tokenizer.pad_token_id
        mask_token_id = self.tokenizer.mask_token_id
        remove = torch.tensor(
            [mask_token_id, pad_token_id], device=pred_ids.device
        )
        non_mask = torch.isin(
            pred_ids, remove, invert=True
        )  # shape: (batch, seq_len)
        if hold_mask is not None:
            non_mask = torch.logical_or(non_mask, hold_mask)
        clean_pred_ids = _remove_tokens(pred_ids, non_mask, pad_token_id)
        # clean_pred_ids = remove_tokens(pred_ids, remove, pad_token_id)
        return clean_pred_ids

    def decode(self, results: Dict) -> Tuple[
        List[str],
        List[str],
        Integer[TT, " batch seq_len"],
        Integer[TT, " batch seq_len"],
        Integer[TT, " batch seq_len"],
    ]:
        x: Integer[TT, " batch seq_len"] = results["x"]
        positions: Integer[TT, " batch seq_len"] = results["positions"]
        attention_mask: Bool[TT, " batch seq_len"] = results["attention_mask"]
        # all tensors are out of order. Sort them based on positions.
        final_positions, sorted_positions_indices = torch.sort(
            positions, dim=-1
        )
        final_x: Integer[TT, " batch seq_len"] = torch.gather(
            x, dim=-1, index=sorted_positions_indices
        )
        prefix_mask = torch.gather(
            (results["token_type_ids"] <= 1),
            dim=-1,
            index=sorted_positions_indices,
        )
        final_x = self.clean_up_pred_ids(final_x, prefix_mask)
        final_attention_mask: Bool[TT, " batch seq_len"] = torch.gather(
            attention_mask, dim=-1, index=sorted_positions_indices
        )
        out_with_spl_tokens: List[str] = self.tokenizer.batch_decode(
            final_x, skip_special_tokens=False
        )
        out: List[str] = self.tokenizer.batch_decode(
            final_x, skip_special_tokens=True
        )
        return (
            out,
            out_with_spl_tokens,
            final_x,
            final_attention_mask,
            final_positions,
        )

    def _update_history(
        self,
        history: List[List[Tuple[str, float, int]]],
        step_results: Dict[str, Any],
        current_step: int,
    ) -> List[List[Tuple[str, float, int]]]:
        """Update generation history."""
        if not self.return_history:
            return history

        if (
            step_results["predict"] is not None
            and step_results["predict"].any().item()
        ):
            decoded_tuple = self.decode(step_results)
            for batch_idx in (
                step_results["predict"].nonzero().flatten().tolist()
            ):
                history[batch_idx].append(
                    (
                        decoded_tuple[0][batch_idx],
                        float(step_results["t"][batch_idx]),
                        current_step,
                    )
                )
        return history

    def to_dict(
        self,
        batch: IdlmBatch,
        preds: IdlmPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convert predictions to dictionary format."""
        preds_list: List[
            Tuple[str, str, List[int], List[Tuple[str, float, int]]]
        ] = list(
            zip(
                preds["text"],
                preds["text_with_spl_tokens"],
                preds["ids"].tolist(),
                preds["history"],
            )
        )
        dicts: List[Dict[str, Any]] = []
        for text, text_with_spl_tokens, ids, history in preds_list:
            rounded_history = [
                [subseq, round(t, 4), step] for subseq, t, step in history
            ]
            dicts.append(
                {
                    "text": text,
                    "text_with_spl_tokens": text_with_spl_tokens,
                    "ids": ids,
                    "history": rounded_history,
                }
            )
        return dicts


class IdlmPredictor(
    torch.nn.Module,
    IdlmPredictorUtilitiesMixin,
    Predictor[IdlmBatch, IdlmPredictionDict],
):
    """IDLM Predictor for generating text using diffusion-based insertion."""

    token_ids_to_suppress: Integer[TT, " n_tokens_to_suppress"]

    def __init__(
        self,
        max_steps: int,
        max_length: int,
        tokenizer: Optional[Tokenizer] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        tokens_to_suppress: Optional[List[str]] = None,
        return_history: bool = False,
        sampling_method: Literal[
            "sample", "sample_top_k", "sample_top_p"
        ] = "sample",
        top: int = 1000,
        p: float = 0.9,
        second_sampling_method: Optional[
            Literal["sample", "sample_top_k", "sample_top_p"]
        ] = None,
        second_top: int = 1000,
        second_p: float = 0.9,
        model: Optional[IdlmModel] = None,
        length_temperature: float = 1.0,
        use_first_step_factor: bool = True,
        send_t_to_model: bool = False,
    ):
        """Initialize the IDLM predictor.

        Args:
            max_steps: Maximum number of generation steps.
            max_length: Maximum sequence length.
            tokenizer: Tokenizer instance.
            noise_schedule: Noise schedule for diffusion.
            tokens_to_suppress: List of token strings to suppress during generation.
            return_history: Whether to return generation history.
            sampling_method: Sampling method for token generation.
            top: Top-k parameter for top-k sampling.
            p: Top-p parameter for nucleus sampling.
            second_sampling_method: Optional second sampling method.
            second_top: Top-k parameter for second sampling method.
            second_p: Top-p parameter for second sampling method.
            model: The IDLM model.
            length_temperature: Temperature for length prediction.
            use_first_step_factor: Whether to use first step factor.
            send_t_to_model: Whether to send time step to model.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required")

        token_ids_to_suppress = torch.tensor(
            [
                tokenizer.convert_tokens_to_ids(token)
                for token in (
                    tokens_to_suppress
                    or [
                        tokenizer.mask_token,
                        tokenizer.eos_token,
                        tokenizer.pad_token,
                        tokenizer.cls_token,
                        tokenizer.bos_token,
                    ]
                )
            ],
            dtype=torch.long,
            requires_grad=False,
        )

        super().__init__()
        self.model = model
        self.register_buffer("token_ids_to_suppress", token_ids_to_suppress)
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_length = max_length

        # Set up sampling functions
        if sampling_method == "sample":
            self.sampling_function = sample_from_logits
        elif sampling_method == "sample_top_k":
            self.sampling_function = partial(sample_from_top_k, top)
        elif sampling_method == "sample_top_p":
            self.sampling_function = partial(sample_from_top_p, p)
        else:
            raise ValueError(f"Invalid sampling method: {sampling_method}")

        if second_sampling_method == "sample":
            self.second_sampling_function = sample_from_logits
        elif second_sampling_method == "sample_top_k":
            self.second_sampling_function = partial(
                sample_from_top_k, second_top
            )
        elif second_sampling_method == "sample_top_p":
            self.second_sampling_function = partial(
                sample_from_top_p, second_p
            )
        elif second_sampling_method is None:
            self.second_sampling_function = None  # type: ignore
        else:
            raise ValueError(
                f"Invalid second sampling method: {second_sampling_method}"
            )

        self.noise_schedule = noise_schedule  # type: ignore
        self.return_history = return_history
        self.length_temperature = length_temperature
        self.use_first_step_factor = use_first_step_factor
        self.send_t_to_model = send_t_to_model

        # Simplified stepping - we'll use a fixed step size
        self.dt = 1.0 / (max_steps + 1)

    def _factor(
        self, n: Integer[TT, " batch"], dot_sigma: Float[TT, " batch"]
    ) -> Float[TT, " batch"]:
        """Compute diffusion factor using series expansion."""
        return incomplete_gamma_factor_using_series(n, dot_sigma)

    def _predict_single_step(
        self,
        step_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Predict a single step of token insertion."""
        x_t: Integer[TT, " batch seq_len"] = step_results["x"]
        positions: Integer[TT, " batch seq_len"] = step_results["positions"]
        attention_mask: Bool[TT, " batch seq_len"] = step_results[
            "attention_mask"
        ]
        t: Float[TT, " batch"] = step_results["t"]
        constraint: Bool[TT, " batch seq_len"] = step_results["constraint"]
        token_type_ids: Integer[TT, " batch seq_len"] = step_results[
            "token_type_ids"
        ]
        cls_position: Integer[TT, " batch"] = step_results["cls_position"]

        # Update time step
        delta_t = self.dt
        s = t - delta_t

        # Get noise parameters
        assert self.noise_schedule is not None
        noise_rate, total_noise = self.noise_schedule(t)

        model = cast(IdlmModel, self.model)
        logits, length_logits = model(
            x_t,
            t if self.send_t_to_model else total_noise,
            attention_mask,
            positions=positions,
            cls_position=cls_position,
        )

        # Suppress specified tokens
        logits[:, :, self.token_ids_to_suppress] = -torch.inf

        # Suppress predictions from constrained positions
        suppress_positions = torch.logical_or(~attention_mask, constraint)
        logits = torch.where(
            suppress_positions.unsqueeze(-1),
            -torch.inf,
            logits,
        )

        # Get mean delta length prediction
        mean_delta_l = model.get_mean_delta_l(
            length_logits, attention_mask, temperature=self.length_temperature
        )

        # Simplified probability calculation
        p = (noise_rate * mean_delta_l / (total_noise + 1e-6)) * delta_t

        # Apply first step factor if needed
        if (
            self.use_first_step_factor
            and not step_results["first_step_done"].all()
        ):
            f = self._factor(mean_delta_l, total_noise)
            f = torch.where(
                step_results["first_step_done"],
                torch.ones_like(f, dtype=p.dtype),
                f,
            )
            p = p / f

        # sample whether to predict or not
        predict = torch.rand_like(p) < p
        predict = torch.logical_and(
            predict, attention_mask.sum(-1) < self.max_length
        )

        step_results["first_step_done"] = step_results[
            "first_step_done"
        ].logical_or(predict)

        # If no predictions, just update time
        if not predict.any().item():
            return {
                "x": x_t,
                "positions": positions,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "t": s,
                "mean_delta_l": mean_delta_l,
                "p": p,
                "length_logits": length_logits,
                "constraint": step_results["constraint"],
                "oracle_length": step_results.get("oracle_length", None),
                "predict": predict,
                "first_step_done": step_results["first_step_done"],
                "cls_position": cls_position,
            }

        # Sample position and token
        pred_seq_index, pred_vocab_index = general_sample_over_last_two_dims(
            logits, self.sampling_function, self.second_sampling_function
        )  # shape (batch,), (batch,)

        # Note: pred_seq_index is not the real index in the token sequence because logits and x_t are kept out of order.
        # Get real position for insertion
        pred_real_index = positions.gather(
            dim=-1, index=pred_seq_index.unsqueeze(-1)
        ).squeeze(-1)

        # Insert tokens only where predict is True else place a pad which we will remove later
        inserted_tokens = torch.where(
            predict,
            pred_vocab_index,
            self.tokenizer.mask_token_id,  # can't insert pads because we may have left-padding, need a different token
        )

        # Concatenate new tokens
        x_s = torch.cat([x_t, inserted_tokens.unsqueeze(-1)], dim=-1)

        # update attention mask, positions, constraint, token_type_ids, etc.
        # attention mask
        pred_attention_mask = torch.cat(
            [attention_mask, predict.unsqueeze(-1)], dim=-1
        )

        # position: increment by 1 positions that are greater than the inserted position
        pos_greater_than_inserted_position = (
            positions > pred_real_index.unsqueeze(-1)
        )
        pos_greater_than_inserted_position = torch.logical_and(
            pos_greater_than_inserted_position,
            predict.unsqueeze(-1),
        )

        inserted_positions = pred_real_index + 1
        pred_positions = positions + pos_greater_than_inserted_position.to(
            dtype=positions.dtype
        )
        pred_positions = torch.cat(
            [pred_positions, inserted_positions.unsqueeze(-1)], dim=-1
        )

        # Update constraint and token type ids
        constraint = torch.cat(
            [
                step_results["constraint"],
                torch.zeros(
                    (pred_positions.shape[0], 1),
                    device=step_results["constraint"].device,
                    dtype=torch.bool,
                ),
            ],
            dim=-1,
        )

        token_type_ids = torch.cat(
            [
                token_type_ids,
                torch.full(
                    (token_type_ids.shape[0], 1),
                    2,  # non-prefix token type id
                    device=token_type_ids.device,
                    dtype=token_type_ids.dtype,
                ),
            ],
            dim=-1,
        )

        return {
            "x": x_s,
            "positions": pred_positions,
            "attention_mask": pred_attention_mask,
            "token_type_ids": token_type_ids,
            "t": s,
            "mean_delta_l": mean_delta_l,
            "p": p,
            "length_logits": length_logits,
            "constraint": constraint,
            "oracle_length": step_results.get("oracle_length", None),
            "predict": predict,
            "first_step_done": step_results["first_step_done"],
            "cls_position": cls_position,
        }

    def _stop(
        self,
        step_results: Dict[str, Any],
        current_step: int,
    ) -> bool:
        """Check if generation should stop."""
        t = step_results["t"]
        attention_mask = step_results["attention_mask"]
        current_length = attention_mask.sum(dim=-1)

        time_ended = not bool((t > 0).any())
        max_steps_reached = current_step > self.max_steps
        max_length_reached = bool((current_length >= self.max_length).all())

        oracle_length = step_results.get("oracle_length", None)
        if oracle_length is not None:
            max_length_reached = bool((current_length >= oracle_length).all())

        return time_ended or max_steps_reached or max_length_reached

    @torch._dynamo.disable()
    def predict(
        self,
        batch: IdlmBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> IdlmPredictionDict:
        """Generate text using IDLM diffusion-based insertion."""
        t = batch["t"]
        if t is None:
            raise NotImplementedError("Timestep determination not implemented")

        if isinstance(t, int):
            t_: Float[TT, " batch"] = torch.full(
                (batch["input_ids"].shape[0],),
                float(t),
                device=batch["input_ids"].device,
            )
        else:
            t_ = t.float()

        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        positions = attention_mask.cumsum(dim=-1) - 1

        if batch["constraint"] is not None:
            constraint = batch["constraint"]
        else:
            # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
            constraint = batch["token_type_ids"] == 0

        step_results = {
            "x": batch["input_ids"],
            "positions": positions,
            "attention_mask": attention_mask,
            "token_type_ids": batch["token_type_ids"],
            "t": t_,
            "mean_delta_l": None,
            "p": None,
            "length_logits": None,
            "constraint": constraint,
            "oracle_length": batch.get("oracle_length", None),
            "predict": torch.ones_like(t_, dtype=torch.bool),
            "first_step_done": torch.zeros_like(t_, dtype=torch.bool),
            "cls_position": batch["cls_position"],
        }

        current_step = 1
        history: List[List[Tuple[str, float, int]]] = [
            [] for _ in range(batch["input_ids"].shape[0])
        ]
        history = self._update_history(history, step_results, current_step)

        # Generation loop
        while not self._stop(step_results, current_step):
            step_results = self._predict_single_step(step_results)
            history = self._update_history(history, step_results, current_step)
            current_step += 1

        # Final step
        step_results["t"] = self.dt * torch.ones_like(t_)
        step_results = self._predict_single_step(step_results)
        history = self._update_history(history, step_results, current_step)

        # Decode final results
        (
            out,
            out_with_spl_tokens,
            final_x,
            final_attention_mask,
            final_positions,
        ) = self.decode(step_results)

        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": final_x,
            "attention_mask": final_attention_mask,
            "positions": final_positions,
            "history": history,
            "loss": None,
        }

    @torch.inference_mode()
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate text from prompts."""
        # Convert prompts to input batch
        token_ids = self.tokenizer(prompts, add_special_tokens=False)[
            "input_ids"
        ]

        # Use the collator helper to prepare the batch properly
        batch_dict = prepare_prefix_ids_idlm(
            token_ids,
            self.tokenizer.pad_token_id,
            max_seq_len=None,  # Let it determine based on content
            cls_token_id=self.tokenizer.cls_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            bos_side="left",  # Put BOS on the left like ILM
        )

        # Add the required IDLM-specific fields
        batch_size = len(prompts)
        device = next(self.parameters()).device

        # Move tensors to device and add missing fields
        batch = {
            "input_ids": batch_dict["input_ids"].to(device),
            "attention_mask": batch_dict["attention_mask"].to(device),
            "token_type_ids": batch_dict["token_type_ids"].to(device),
            "t": torch.ones(
                batch_size, device=device, dtype=torch.float
            ),  # Start from t=1
            "constraint": None,
            "noise_rate": torch.ones(
                batch_size, device=device, dtype=torch.float
            )
            * 0.1,
            "total_noise": torch.ones(
                batch_size, device=device, dtype=torch.float
            ),
            "n_drops": None,
            "target_ids": None,
        }

        preds = self.predict(batch)
        return preds["text"]
