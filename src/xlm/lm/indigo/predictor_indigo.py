from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
    Literal,
    Union,
)
from functools import partial
import torch
from jaxtyping import Bool, Integer
from xlm.datamodule import Tokenizer
from torch import Tensor as TT
from xlm.harness import Predictor
from xlm.noise import NoiseSchedule
from .types_indigo import (
    IndigoBatch,
    IndigoPredictionDict,
    IndigoSeq2SeqPredBatch,
    IndigoPredBatch,
    IndigoModelProtocol,
)
from xlm.utils.nn import (
    sample_from_logits,
    sample_from_top_k,
    sample_from_top_p,
)

import time


###############################################################
# region: Predictors

# import functions from utils
from xlm.lm.indigo.utils import (
    get_tertiary_relative_position_matrix,
)


class IndigoPredictor(
    torch.nn.Module,
    Predictor[IndigoBatch, IndigoPredictionDict],
):

    def __init__(
        self,
        max_steps: int,
        max_length: int,
        tokenizer: Optional[Tokenizer] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        model: Optional[IndigoModelProtocol] = None,
        sampling_method: Literal[
            "sample", "sample_top_k", "sample_top_p"
        ] = "sample",
        top: Optional[int] = None,
        position_sampling_method: Literal[
            "sample", "sample_top_k", "sample_top_p"
        ] = "sample",
        position_top: Optional[int] = None,
    ):
        """Constructor for IndigoPredictor.

        Args:
            max_steps: Maximum number of prediction steps.
            max_length: Maximum sequence length.
            tokenizer: The tokenizer to use.
            noise_schedule: Noise schedule (not used in Indigo but kept for interface consistency).
            model: The Indigo model to use for predictions.
            sampling_method: Sampling method to use.
            top: Top-p or top-k parameter for sampling depending on the selected sampling_method.
            position_sampling_method: Sampling method to use for position sampling.
            position_top: Top-p or top-k parameter for position sampling depending on the selected position_sampling_method.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        super().__init__()
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_length = max_length
        self.noise_schedule = noise_schedule
        self.model = model
        if sampling_method == "sample":
            self.sampling_function = sample_from_logits
        elif sampling_method == "sample_top_k":
            if top is None:
                raise ValueError(
                    "top is required when sampling_method is sample_top_k"
                )
            self.sampling_function = partial(sample_from_top_k, top)
        elif sampling_method == "sample_top_p":
            if top is None:
                raise ValueError(
                    "top is required when sampling_method is sample_top_p"
                )
            self.sampling_function = partial(sample_from_top_p, top)
        else:
            raise ValueError(f"Invalid sampling method: {sampling_method}")

        if position_sampling_method == "sample":
            self.position_sampling_function = sample_from_logits
        elif position_sampling_method == "sample_top_k":
            if position_top is None:
                raise ValueError(
                    "position_top is required when position_sampling_method is sample_top_k"
                )
            self.position_sampling_function = partial(
                sample_from_top_k, position_top
            )
        elif position_sampling_method == "sample_top_p":
            if position_top is None:
                raise ValueError(
                    "position_top is required when position_sampling_method is sample_top_p"
                )
            self.position_sampling_function = partial(
                sample_from_top_p, position_top
            )
        else:
            raise ValueError(
                f"Invalid position sampling method: {position_sampling_method}"
            )

    @torch._dynamo.disable()
    def predict(
        self,
        batch: Union[IndigoSeq2SeqPredBatch, IndigoPredBatch],  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        # fmt: off
        attention_mask = batch['attention_mask'].to(dtype=torch.bool)
        input_ids = batch['input_ids']
        # fmt: on
        # create increasing pi for the input ids
        # Case 1: Unconditional prediction: input_ids are all BOS EOS
        # Case 2: there is prompt: input_ids are:
        # a) left_pads promt_ids BOS EOS
        # b) left_pads BOS prompt EOS
        # In all the above cases pi should simply be increasing
        pi = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        # TODO: Support contraints?

        state: Dict[str, TT] = {
            "x": input_ids,
            "pi": pi,
            "attention_mask": attention_mask.bool(),
        }

        step = 0
        while not self.stop(state, step):
            state = self.predict_single_step(state)
            step += 1

        texts, texts_with, final_ids, final_mask, final_pos = self.decode(
            state
        )

        return {
            "text": texts,
            "text_w_token": texts_with,
            "ids": final_ids,
            "attention_mask": final_mask,
            "positions": final_pos,
            "loss": None,
            "time_taken": [time.time() - t0] * len(texts),
        }

    @torch.no_grad()
    def predict_single_step(self, state: Dict[str, TT]) -> Dict[str, TT]:
        """
        Single step prediction for Indigo.
        """
        # fmt: off
        x = state['x']
        pi = state['pi']
        attention_mask = state['attention_mask']
        finished = state['finished']
        # fmt: on
        rel_matrix = get_tertiary_relative_position_matrix(pi)
        assert self.model is not None
        model = cast(IndigoModelProtocol, self.model)
        hidden_states, vocab_logits = model(x, rel_matrix, attention_mask)

        next_token_logits = vocab_logits[:, -1, :]
        # TODO: suppress special tokens?
        next_token = self.sampling_function(
            next_token_logits
        )  # shape (batch,)
        # if allready finished, set predicted token to pad
        next_token = next_token.where(
            finished.unsqueeze(-1), self.tokenizer.pad_token_id
        )
        finished = finished.logical_or(
            next_token == self.tokenizer.cls_token_id
        )  # cls is EOD
        position_logits = model.get_position_logits(
            hidden_states, next_token.unsqueeze(-1)
        ).squeeze(
            -1
        )  # shape (batch, key_seq_len, 2)
        # TODO (constraints): Can add constraints here
        # construct unnormalized log-probabilities i.e, logits for VALID slots
        # a valid slot is a pair (i, right_neighbor_of_i)
        is_on_right = pi.unsqueeze(-1) < pi.unsqueeze(
            -2
        )  # shape (batch, key_seq_len, query_seq_len)
        # is_on_right[*, i, j] = 1 if pi[i] < pi[j]
        right_neighbor_idx, right_neighbor_abs_pos = torch.min(
            is_on_right * pi.unsqueeze(-1), dim=-1
        )  # shape (batch, query_seq_len)
        slot_left_logits = position_logits[
            :, :, 0
        ]  # shape (batch, key_seq_len)
        slot_right_logits = position_logits[:, :, 1].gather(
            -1, right_neighbor_idx
        )
        # suppress slots that are right of EOS
        # get the absolute position of EOS
        eos_idx = (
            (x == self.tokenizer.eos_token_id).int().argmax(dim=-1)
        )  # shape (batch,)
        after_eos = (
            pi.gather(-1, eos_idx.unsqueeze(-1)) < pi
        )  # shape (batch, seq_len)
        # unnormalized log-probabilities, ie logits. Finally!
        lae = torch.logaddexp(
            slot_left_logits, slot_right_logits
        )  # shape (batch, key_seq_len, 1)
        lae.masked_fill_(after_eos, -torch.inf)  # shape (batch, key_seq_len)
        sampled_slot = self.position_sampling_function(lae)  # shape (batch,)
        # extend x, pi, and attention_mask
        # pi
        sampled_abs_pos = pi.gather(
            -1, sampled_slot.unsqueeze(-1)
        )  # shape (batch,1)
        # for finished sequences, set sampled_abs_pos to max(pi) + 1
        sampled_abs_pos = sampled_abs_pos.where(
            finished.unsqueeze(-1), pi.max(dim=-1, keepdim=True).values
        )
        # everything to the right of sampled_slot is shifted by +1
        pi = pi.where(sampled_abs_pos < pi, pi + 1)
        pi = torch.cat([pi, sampled_abs_pos + 1], dim=-1)
        x = torch.cat([x, next_token.unsqueeze(-1)], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, ~finished.unsqueeze(-1)], dim=-1
        )
        return {
            "x": x,
            "pi": pi,
            "attention_mask": attention_mask,
            "finished": finished,
        }

    def stop(self, state: Dict[str, TT], step: int) -> bool:
        if step >= self.max_steps:
            return True
        if torch.all(state["finished"]):
            return True
        if torch.all(state["attention_mask"].sum(dim=1) >= self.max_length):
            return True
        return False

    def decode(self, state: Dict[str, TT]) -> Tuple[
        List[str],
        List[str],
        Integer[TT, "B L"],
        Bool[TT, "B L"],
        Integer[TT, "B L"],
    ]:
        """
        Sort tokens by absolute positions, strip specials, and decode.
        Returns (clean text, text with specials, ids, mask, positions) in sorted order.
        """
        x = state["x"]
        pos = state["pi"]
        amask = state["attention_mask"]

        # reorder by absolute positions
        sorted_pos, sort_idx = torch.sort(pos, dim=-1)
        x_sorted = torch.gather(x, -1, sort_idx)
        mask_sorted = torch.gather(amask, -1, sort_idx)
        # No need for removing special tokens
        out = self.tokenizer.batch_decode(x_sorted, skip_special_tokens=True)
        out_with_spl_tokens = self.tokenizer.batch_decode(
            x_sorted, skip_special_tokens=False
        )
        return out, out_with_spl_tokens, x_sorted, mask_sorted, sorted_pos


# endregion: Predictors
###############################################################
