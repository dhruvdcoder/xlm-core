from typing import Any, Dict, List, Optional, Tuple, Literal, TypedDict
from functools import partial
import torch
from jaxtyping import Bool, Integer, Float
from xlm.datamodule import Tokenizer
from torch import Tensor as TT
from xlm.harness import Predictor
from .types import ZLMModel, ZLMPredictionDict
from xlm.noise import NoiseSchedule

from xlm.utils.nn import (
    sample_from_logits,
    sample_from_top_k,
    sample_from_top_p,
)


###############################################################
# region: Types


class ZLMStepResults(TypedDict):
    """Step results for ZLM prediction.

    Attributes:
        x: Integer[TT, " batch seq_len"] Current predicted sequence.
        attention_mask: Bool[TT, " batch seq_len"] Mask of the current sequence.
        logits: Float[TT, " batch seq_len vocab_size"] Logits of the current sequence.
    """

    x: Integer[TT, " batch seq_len"]
    attention_mask: Bool[TT, " batch seq_len"]
    logits: Optional[Float[TT, " batch seq_len vocab_size"]]


# endregion: Types
###############################################################


###############################################################
# region: Predictors


class ZLMPredictor(
    torch.nn.Module,
    Predictor[Dict[str, Any], ZLMPredictionDict],
):

    def __init__(
        self,
        max_steps: int,
        max_length: int,
        tokenizer: Optional[Tokenizer] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        sampling_method: Literal[
            "sample", "sample_top_k", "sample_top_p"
        ] = "sample",
        top: int = 1000,
        p: float = 0.9,
        model: Optional[ZLMModel] = None,
    ):
        """Constructor for ZLMPredictor.

        Args:
            max_steps: Maximum number of prediction steps.
            max_length: Maximum sequence length.
            tokenizer: The tokenizer to use.
            noise_schedule: Noise schedule (not used in ZLM but kept for interface consistency).
            sampling_method: Sampling method to use.
            top: Top-k parameter for sampling.
            p: Top-p parameter for sampling.
            model: The ZLM model to use for predictions.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        super().__init__()
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_length = max_length
        self.noise_schedule = noise_schedule

        if sampling_method == "sample":
            self.sampling_function = sample_from_logits
        elif sampling_method == "sample_top_k":
            self.sampling_function = partial(sample_from_top_k, top)
        elif sampling_method == "sample_top_p":
            self.sampling_function = partial(sample_from_top_p, p)
        else:
            raise ValueError(f"Invalid sampling method: {sampling_method}")
        self.model = model

    def predict_single_step(
        self,
        step_results: ZLMStepResults,
        current_step: int,
    ) -> ZLMStepResults:
        """Predict the next token in the sequence.

        Args:
            step_results: Current step results containing x, attention_mask, and logits.
            current_step: Current prediction step.
            final_step: Whether this is the final step.

        Returns:
            Updated step results with the next token predicted.
        """
        x: Integer[TT, " batch seq_len"] = step_results["x"]
        attention_mask: Bool[TT, " batch seq_len"] = step_results[
            "attention_mask"
        ]

        # Create causal attention mask
        _, seq_len = attention_mask.shape
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=attention_mask.device,
            )
        )

        # Expand attention_mask and combine with causal mask
        expanded_attention_mask = attention_mask.unsqueeze(1)
        causal_attention_mask = expanded_attention_mask & causal_mask

        # Create position IDs
        positions = attention_mask.cumsum(dim=1) - 1
        positions *= attention_mask

        # Get logits from the model
        assert self.model is not None
        logits = self.model(x, causal_attention_mask, positions)

        # Get logits for the last position (next token prediction)
        logits = logits[:, -1, :]

        # Sample the next token
        x_pred = self.sampling_function(logits)  # (batch, )

        x_pred = x_pred.unsqueeze(-1)  # (batch, 1)

        # Append only PAD tokens if EOS is reached
        eos_mask = (x == self.tokenizer.eos_token_id).any(
            dim=-1, keepdim=True
        )  # (batch, 1)
        x_pred = torch.where(eos_mask, self.tokenizer.pad_token_id, x_pred)

        x = torch.cat([x, x_pred], dim=-1)  # (batch, seq_len + 1)

        # Update attention mask
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    1,
                    device=attention_mask.device,
                    dtype=torch.bool,
                ),
            ],
            dim=-1,
        )

        return {
            "x": x,
            "attention_mask": attention_mask,
            "logits": logits,
        }

    def stop(
        self,
        step_results: ZLMStepResults,
        current_length: int,
    ) -> bool:
        """Check if prediction should stop.

        Args:
            step_results: Current step results.
            current_length: Current sequence length.

        Returns:
            True if prediction should stop, False otherwise.
        """
        x = step_results["x"]
        max_length_reached = current_length >= self.max_length
        is_eos_reached = (x == self.tokenizer.eos_token_id).any(dim=1).all()
        return bool(max_length_reached) or bool(is_eos_reached)

    def decode(self, results: ZLMStepResults) -> Tuple[
        List[str],
        List[str],
        Integer[TT, " batch seq_len"],
    ]:
        """Decode the predicted sequence.

        Args:
            results: Step results containing the predicted sequence.

        Returns:
            Tuple of (decoded_text, decoded_text_with_special_tokens, token_ids).
        """
        x: Integer[TT, " batch seq_len"] = results["x"]
        out = self.tokenizer.batch_decode(x, skip_special_tokens=True)
        out_with_spl_tokens: List[str] = self.tokenizer.batch_decode(
            x, skip_special_tokens=False
        )
        return out, out_with_spl_tokens, x

    @torch._dynamo.disable()
    def predict(
        self,
        batch: Dict[str, Any],  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
        max_len: int = 0,
    ) -> ZLMPredictionDict:
        """Predict the complete sequence.

        Args:
            batch: Input batch containing input_ids and attention_mask.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.
            dataloader_name: Dataloader name.
            max_len: Maximum length for prediction.

        Returns:
            Prediction results containing text, token IDs, and attention mask.
        """
        step_results: ZLMStepResults = {
            "x": batch[
                "input_ids"
            ],  # don't clone assuming that the caller is prepared for in-place operations
            "attention_mask": batch["attention_mask"],
            "logits": None,  # type: ignore ok for first step
        }

        # assume left-padded input_ids
        output_start_idx = batch["input_ids"].size(1)
        current_length = batch["input_ids"].size(1)

        # Generate tokens step by step
        while not self.stop(step_results, current_length):
            step_results = self.predict_single_step(
                step_results, current_length
            )
            current_length += 1

        # PAD to max_length
        step_results["x"] = torch.nn.functional.pad(
            step_results["x"],
            (0, self.max_length - step_results["x"].size(1)),
            value=self.tokenizer.pad_token_id,
        )
        step_results["attention_mask"] = torch.nn.functional.pad(
            step_results["attention_mask"],
            (0, self.max_length - step_results["attention_mask"].size(1)),
            value=False,
        )
        step_results["logits"] = None

        # Decode the final result
        (
            out,
            out_with_spl_tokens,
            final_x,
        ) = self.decode(step_results)

        # Create position IDs for the final sequence
        positions = step_results["attention_mask"].cumsum(dim=1) - 1
        positions *= step_results["attention_mask"]

        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": final_x,
            "attention_mask": step_results["attention_mask"],
            "positions": positions,
            "time_taken": [0.0] * len(out),  # Placeholder for timing
            "output_start_idx": output_start_idx,
        }

    def to_dict(
        self,
        batch: Dict[str, Any],  # type: ignore
        preds: ZLMPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convert predictions to dictionary format.

        Args:
            batch: Input batch.
            preds: Prediction results.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.
            dataloader_name: Dataloader name.

        Returns:
            List of dictionaries containing prediction results.
        """
        preds_list: List[Tuple[str, str, List[int]]] = list(
            zip(
                preds["text"],
                preds["text_with_spl_tokens"],
                preds["ids"].tolist(),
            )
        )

        dicts: List[Dict[str, Any]] = []
        for text, text_with_spl_tokens, ids in preds_list:
            dicts.append(
                {
                    "text": text,
                    "text_with_spl_tokens": text_with_spl_tokens,
                    "ids": ids,
                }
            )
        return dicts


# endregion: Predictors
###############################################################
