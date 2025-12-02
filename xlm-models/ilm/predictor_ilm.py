from typing import Any, Dict, List, Optional, Tuple, cast, Literal, Callable
from itertools import cycle
from functools import partial
import torch
from jaxtyping import Bool, Integer
from xlm import flags
from xlm.datamodule import Tokenizer
from torch import Tensor as TT
from .types_ilm import (
    ILMBatch,
    ILMPredictionDict,
    ILMModel,
    ILMSeq2SeqPredictionBatch,
)
from xlm.harness import Predictor
from xlm.noise import NoiseSchedule
from .datamodule_ilm import prepare_prefix_ids, print_batch_ilm
from .nn import (
    _remove_tokens,
    remove_tokens,
    general_sample_over_last_two_dims,
)
from xlm.utils.nn import (
    sample_from_logits,
    sample_from_top_k,
    sample_from_top_p,
)
import time


###############################################################
# region: Predictors


class ILMPredictorUtilitiesMixin:
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
                        1.0,
                        current_step,
                    )
                )
        return history

    def to_dict(
        self,
        batch: ILMBatch,  # type: ignore
        preds: ILMPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        preds_list: List[
            Tuple[str, str, List[int], List[Tuple[str, float, int]], float]
        ] = list(
            zip(
                preds["text"],
                preds["text_with_spl_tokens"],
                preds["ids"].tolist(),
                preds["history"],
                preds.get(
                    "time_taken", cycle([-1])
                ),  # -1 when the predict method does not measure time.
            )
        )
        dicts: List[Dict[str, Any]] = []
        for text, text_with_spl_tokens, ids, history, time_taken in preds_list:
            rounded_history = [
                [subseq, round(t, 4), step] for subseq, t, step in history
            ]
            dicts.append(
                {
                    "text": text,
                    "text_with_spl_tokens": text_with_spl_tokens,
                    "ids": ids,
                    "history": rounded_history,
                    "time_taken": time_taken,
                }
            )
        return dicts


class ILMPredictor(
    torch.nn.Module,
    ILMPredictorUtilitiesMixin,
    Predictor[ILMBatch, ILMPredictionDict],
):
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
        model: Optional[ILMModel] = None,
        input_constraint: bool = False,
    ):
        """Constructor for ILMPredictor.

        Args:
            max_steps (int): The maximum number of steps to take.
            max_length (int): The maximum length (excluding special tokens like PAD and MASK)
                of the generated text.
            stopping_threshold (float): The threshold for stopping use on the length classification scores.
            tokenizer (Tokenizer): The tokenizer. Typically, set after initialization but before calling predict.
            noise_schedule (NoiseSchedule): The noise schedule. Typically, set after initialization but before calling predict.
            tokens_to_suppress (List[str]): The tokens to suppress during generation.
            return_history (bool): Whether to return the history.
            sampling_method (Literal["sample", "sample_top_k", "sample_top_p"]): The sampling method.
                When `second_sampling_method` is not provided, the specified method here is
                used to sample from the joint distribution of positions and tokens.
                When `second_sampling_method` is provided, the specified method here is
                used to sample from the token distribution (conditional) given the postions sampled
                using the `second_sampling_method`.
                "sample" means vanilla sampling from the distribution.
                "sample_top_k" means sampling from the top-k distribution.
                "sample_top_p" means sampling from the top-p distribution (neuclius samplingn).
            top (int): The top-k sampling parameter for `sampling_method`.
            p (float): The top-p sampling parameter for `sampling_method`.
            second_sampling_method (Optional[Literal["sample", "sample_top_k", "sample_top_p"]]): The second sampling method.
            second_top (int): The second top-k sampling parameter for `second_sampling_method`.
            second_p (float): The second top-p sampling parameter for `second_sampling_method`.
            model (Optional[ILMModel]): The model. Typically, set after initialization but before calling predict.
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
                        # tokenizer.pad_token, # don't suppress pad tokens
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
        self.register_buffer(
            "token_ids_to_suppress", token_ids_to_suppress, persistent=False
        )
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_length = max_length
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
            self.second_sampling_function = None
        else:
            raise ValueError(
                f"Invalid second sampling method: {second_sampling_method}"
            )

        self.noise_schedule = noise_schedule
        self.return_history = return_history
        self.input_constraint = input_constraint

    def _predict_single_step(
        self,
        step_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        TODO (doc): Add docstring.
        Constraints:
            - Mask tokens cannot be predicted
            - Input non-mask tokens cannot be changed
        """
        # fmt: off
        x_t: Integer[TT, " *batch seq_len"] = step_results["x"]
        positions: Integer[TT, " *batch seq_len"] = step_results["positions"]
        attention_mask: Bool[TT, " *batch seq_len"] = step_results["attention_mask"]
        constraint: Bool[TT, " *batch seq_len"] = step_results["constraint"]
        token_type_ids: Integer[TT, " *batch seq_len"] = step_results["token_type_ids"]
        # fmt: on
        model = cast(ILMModel, self.model)
        logits, _ = model(
            x_t,
            attention_mask,
            positions=positions,
            token_type_ids=constraint if self.input_constraint else None,
        )

        # suppress some specified (mostly special) tokens
        logits[:, :, self.token_ids_to_suppress] = -torch.inf
        # suppress predictions from input tokens that are mask or pad or part of the prefix
        suppress_positions = torch.logical_or(~attention_mask, constraint)
        logits = torch.where(
            suppress_positions.unsqueeze(-1),
            -torch.inf,
            logits,
        )
        # make sure that predict never goes from False to True
        pred_seq_index, pred_vocab_index = general_sample_over_last_two_dims(
            logits, self.sampling_function, self.second_sampling_function
        )  # shape (batch,), (batch,)
        # pred_seq_index is not the real index in the token sequence because logits and x_t are kept out of order.
        pred_real_index = positions.gather(
            dim=-1, index=pred_seq_index.unsqueeze(-1)
        ).squeeze(
            -1
        )  # shape (batch,)
        inserted_tokens = pred_vocab_index
        # predict = step_results["predict"]
        current_length = attention_mask.sum(dim=-1)
        predict = current_length < self.max_length
        x_s = torch.cat(
            [x_t, inserted_tokens.unsqueeze(-1)], dim=-1
        )  # shape (*batch, current_seq_len + 1)
        pred_attention_mask = torch.cat(
            [attention_mask, predict.unsqueeze(-1)], dim=-1
        )  # shape (*batch, current_seq_len + 1)
        pos_greater_than_inserted_position = (
            positions > pred_real_index.unsqueeze(-1)
        )  # mask of shape (batch, current_seq_len)
        pos_greater_than_inserted_position = torch.logical_and(
            pos_greater_than_inserted_position,
            predict.unsqueeze(-1),
        )
        inserted_positions = pred_real_index + 1
        # increment positions greater than inserted position
        pred_positions = positions + pos_greater_than_inserted_position.to(
            dtype=positions.dtype
        )
        pred_positions = torch.cat(
            [pred_positions, inserted_positions.unsqueeze(-1)], dim=-1
        )  # shape (*batch, current_seq_len + 1)
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

        step_result = {
            "x": x_s,
            "positions": pred_positions,
            "attention_mask": pred_attention_mask,
            "token_type_ids": token_type_ids,
            "constraint": constraint,
            "predict": predict,
        }
        return step_result

    def _stop(
        self,
        step_results: Dict[str, Any],
        current_step: int,
    ) -> bool:

        max_steps_reached = current_step >= self.max_steps
        return max_steps_reached

    @torch._dynamo.disable()
    def predict(
        self,
        batch: ILMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMPredictionDict:
        _start_time = time.time()
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        positions = attention_mask.cumsum(dim=-1) - 1
        if batch["constraint"] is not None:
            constraint = batch["constraint"]
        else:
            # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
            constraint = batch["token_type_ids"] <= 1
        current_length = attention_mask.sum(dim=-1)
        predict = current_length < self.max_length
        step_results = {
            "x": batch["input_ids"],
            "positions": positions,
            "attention_mask": attention_mask,
            "token_type_ids": batch["token_type_ids"],
            "length_logits": None,
            "constraint": constraint,
            "oracle_length": batch.get("oracle_length", None),
            "predict": predict,
        }
        current_step = 1
        history: List[List[Tuple[str, float, int]]] = [
            [] for _ in range(batch["input_ids"].shape[0])
        ]
        history = self._update_history(history, step_results, current_step)
        while not self._stop(step_results, current_step):
            step_results = self._predict_single_step(
                step_results,
            )
            history = self._update_history(history, step_results, current_step)
            if flags.DEBUG_PRINT_PREDS:
                print("--" * 10 + f"STEP {current_step}" + "--" * 10)
                print(self.decode(step_results)[0])
            current_step += 1
        # final step (nothing special)
        step_results = self._predict_single_step(
            step_results,
        )
        history = self._update_history(history, step_results, current_step)
        # decode the final step
        (
            out,
            out_with_spl_tokens,
            final_x,
            final_attention_mask,
            final_positions,
        ) = self.decode(step_results)
        _end_time = time.time()
        _time_taken = _end_time - _start_time
        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": final_x,
            "attention_mask": final_attention_mask,
            "positions": final_positions,
            "history": history,
            "loss": None,
            "time_taken": (
                [_time_taken] * len(out)
            ),  # cannot separate time for each sample
        }

    @torch.inference_mode()
    def generate(self, prompts: List[str]) -> List[str]:
        # Convert prompts to input batch
        token_ids = self.tokenizer(prompts, add_special_tokens=False)[
            "input_ids"
        ]

        batch = cast(
            ILMSeq2SeqPredictionBatch,
            prepare_prefix_ids(
                token_ids,
                self.tokenizer.pad_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.bos_token_id,
                bos_side="left",
            ),
        )
        batch["target_ids"] = None
        constraint = torch.ones_like(batch["attention_mask"], dtype=torch.bool)
        constraint[:, -1] = False
        batch["constraint"] = constraint
        # move to device
        batch = {
            k: v.to("cuda")
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
        preds = self.predict(batch)
        return preds["text"]


class ILMPredictorWithLengthClassification(
    torch.nn.Module,
    ILMPredictorUtilitiesMixin,
    Predictor[ILMBatch, ILMPredictionDict],
):
    token_ids_to_suppress: Integer[TT, " n_tokens_to_suppress"]

    def __init__(
        self,
        max_steps: int,
        max_length: int,
        stopping_threshold: float = 0.5,
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
        model: Optional[ILMModel] = None,
        force_predict_first_step: bool = False,
        input_constraint: bool = False,
        use_high_precision: bool = False,
        stopping_temperature: float = 1.0,
    ):
        """Constructor for ILMPredictor.

        Args:
            max_steps (int): The maximum number of steps to take.
            max_length (int): The maximum length (excluding special tokens like PAD and MASK)
                of the generated text.
            stopping_threshold (float): The threshold for stopping use on the length classification scores.
            tokenizer (Tokenizer): The tokenizer. Typically, set after initialization but before calling predict.
            noise_schedule (NoiseSchedule): The noise schedule. Typically, set after initialization but before calling predict.
            tokens_to_suppress (List[str]): The tokens to suppress during generation.
            return_history (bool): Whether to return the history.
            sampling_method (Literal["sample", "sample_top_k", "sample_top_p"]): The sampling method.
                When `second_sampling_method` is not provided, the specified method here is
                used to sample from the joint distribution of positions and tokens.
                When `second_sampling_method` is provided, the specified method here is
                used to sample from the token distribution (conditional) given the postions sampled
                using the `second_sampling_method`.
                "sample" means vanilla sampling from the distribution.
                "sample_top_k" means sampling from the top-k distribution.
                "sample_top_p" means sampling from the top-p distribution (neuclius samplingn).
            top (int): The top-k sampling parameter for `sampling_method`.
            p (float): The top-p sampling parameter for `sampling_method`.
            second_sampling_method (Optional[Literal["sample", "sample_top_k", "sample_top_p"]]): The second sampling method.
            second_top (int): The second top-k sampling parameter for `second_sampling_method`.
            second_p (float): The second top-p sampling parameter for `second_sampling_method`.
            model (Optional[ILMModel]): The model. Typically, set after initialization but before calling predict.
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
        self.stopping_threshold = stopping_threshold
        self.model = model
        self.register_buffer(
            "token_ids_to_suppress", token_ids_to_suppress, persistent=False
        )
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_length = max_length
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
            self.second_sampling_function = None
        else:
            raise ValueError(
                f"Invalid second sampling method: {second_sampling_method}"
            )

        self.noise_schedule = noise_schedule
        self.return_history = return_history
        self.force_predict_first_step = force_predict_first_step
        self.input_constraint = input_constraint
        self.use_high_precision = use_high_precision
        self.stopping_temperature = stopping_temperature

    def _compute_stopping_mask(
        self,
        step_results: Dict[str, Any],
    ) -> Bool[TT, " batch"]:
        length_logits = step_results["length_logits"]
        if length_logits is not None:  # graph break?
            attention_mask = step_results["attention_mask"]
            p = torch.softmax(length_logits / self.stopping_temperature, dim=-1)[
                :, 0
            ]
            predict = p < self.stopping_threshold
            predict.logical_and_(attention_mask.sum(-1) < self.max_length)
        else:
            predict = step_results["predict"]
        oracle_length = step_results.get("oracle_length", None)
        if oracle_length is not None:
            # don't do this update in-place
            predict = predict.logical_and(
                attention_mask.sum(-1) < oracle_length
            )
        return predict

    def _predict_single_step(
        self,
        step_results: Dict[str, Any],
        current_step: int,
    ) -> Dict[str, Any]:
        """
        TODO (doc): Add docstring.
        Constraints:
            - Mask tokens cannot be predicted
            - Input non-mask tokens cannot be changed
        """
        # fmt: off
        x_t: Integer[TT, " *batch seq_len"] = step_results["x"]
        positions: Integer[TT, " *batch seq_len"] = step_results["positions"]
        attention_mask: Bool[TT, " *batch seq_len"] = step_results["attention_mask"]
        constraint: Bool[TT, " *batch seq_len"] = step_results["constraint"]
        token_type_ids: Integer[TT, " *batch seq_len"] = step_results["token_type_ids"]
        cls_position: Integer[TT, " *batch"] = step_results["cls_position"]
        # fmt: on
        model = cast(ILMModel, self.model)
        logits, length_logits = model(
            x_t,
            attention_mask,
            positions=positions,
            token_type_ids=constraint if self.input_constraint else None,
            cls_position=cls_position,
        )

        # suppress some specified (mostly special) tokens
        logits[:, :, self.token_ids_to_suppress] = -torch.inf
        # suppress predictions from input tokens that are mask or pad or part of the prefix
        suppress_positions = torch.logical_or(~attention_mask, constraint)
        logits = torch.where(
            suppress_positions.unsqueeze(-1),
            -torch.inf,
            logits,
        )
        step_results["length_logits"] = length_logits
        predict = self._compute_stopping_mask(step_results)
        if current_step == 1 and self.force_predict_first_step:
            predict = torch.ones_like(predict)
        # make sure that predict never goes from False to True
        if flags.DEBUG_OVERFIT:
            if step_results["predict"] is not None:
                assert not bool(
                    torch.logical_and(
                        ~step_results["predict"],
                        predict,
                    ).any()
                ), "predict went from False to True at indices {}".format(
                    (
                        torch.logical_and(
                            ~step_results["predict"],
                            predict,
                        ).nonzero()
                    )
                )
        else:
            # if predict was false, it should still be false
            predict = predict.logical_and(step_results["predict"])
        if flags.DEBUG_PRINT_PREDS and current_step < 10:
            predict[:] = 1
        if not predict.any().item():
            return {
                "x": x_t,
                "positions": positions,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "length_logits": length_logits,
                "constraint": step_results["constraint"],
                "oracle_length": step_results.get("oracle_length", None),
                "predict": predict,
                "cls_position": cls_position,
            }
        if self.use_high_precision:
            logits = logits.to(dtype=torch.float64)
        pred_seq_index, pred_vocab_index = general_sample_over_last_two_dims(
            logits, self.sampling_function, self.second_sampling_function
        )  # shape (batch,), (batch,)
        # pred_seq_index is not the real index in the token sequence because logits and x_t are kept out of order.
        pred_real_index = positions.gather(
            dim=-1, index=pred_seq_index.unsqueeze(-1)
        ).squeeze(
            -1
        )  # shape (batch,)
        inserted_tokens = torch.where(
            predict,
            pred_vocab_index,
            self.tokenizer.mask_token_id,
        )
        x_s = torch.cat(
            [x_t, inserted_tokens.unsqueeze(-1)], dim=-1
        )  # shape (*batch, current_seq_len + 1)
        pred_attention_mask = torch.cat(
            [attention_mask, predict.unsqueeze(-1)], dim=-1
        )  # shape (*batch, current_seq_len + 1)
        pos_greater_than_inserted_position = (
            positions > pred_real_index.unsqueeze(-1)
        )  # mask of shape (batch, current_seq_len)
        pos_greater_than_inserted_position = torch.logical_and(
            pos_greater_than_inserted_position,
            predict.unsqueeze(-1),
        )
        inserted_positions = pred_real_index + 1
        # increment positions greater than inserted position
        pred_positions = positions + pos_greater_than_inserted_position.to(
            dtype=positions.dtype
        )
        pred_positions = torch.cat(
            [pred_positions, inserted_positions.unsqueeze(-1)], dim=-1
        )  # shape (*batch, current_seq_len + 1)
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

        step_result = {
            "x": x_s,
            "positions": pred_positions,
            "attention_mask": pred_attention_mask,
            "token_type_ids": token_type_ids,
            "length_logits": length_logits,
            "constraint": constraint,
            "oracle_length": step_results.get("oracle_length", None),
            "predict": predict,
            "cls_position": cls_position,
        }
        return step_result

    def _stop(
        self,
        step_results: Dict[str, Any],
        current_step: int,
    ) -> bool:
        max_steps_reached = current_step >= self.max_steps
        batch_done = not bool(step_results["predict"].any().item())
        return max_steps_reached or batch_done

    @torch._dynamo.disable()
    def predict(
        self,
        batch: ILMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMPredictionDict:
        _start_time = time.time()
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        positions = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        constraint = batch.get("constraint")  # expect it to be present
        if constraint is None:
            constraint = batch["token_type_ids"] <= 0

        step_results = {
            "x": batch["input_ids"],
            "positions": positions,
            "attention_mask": attention_mask,
            "token_type_ids": batch["token_type_ids"],
            "length_logits": None,
            "constraint": constraint,
            "oracle_length": batch.get("oracle_length", None),
            "predict": torch.ones(
                (batch["input_ids"].shape[0],),
                dtype=torch.bool,
                device=batch["input_ids"].device,
            ),
            "current_step": 1,
            "cls_position": batch["cls_position"],
        }
        current_step = 1
        history: List[List[Tuple[str, float, int]]] = [
            [] for _ in range(batch["input_ids"].shape[0])
        ]
        history = self._update_history(history, step_results, current_step)
        while not self._stop(step_results, current_step):
            step_results = self._predict_single_step(
                step_results,
                current_step,
            )
            history = self._update_history(history, step_results, current_step)
            if flags.DEBUG_PRINT_PREDS:
                print("--" * 10 + f"STEP {current_step}" + "--" * 10)
                print(self.decode(step_results)[0])
            current_step += 1
            step_results["current_step"] = current_step
        # final step (nothing special)
        step_results = self._predict_single_step(
            step_results,
            current_step,
        )
        history = self._update_history(history, step_results, current_step)
        # decode the final step
        (
            out,
            out_with_spl_tokens,
            final_x,
            final_attention_mask,
            final_positions,
        ) = self.decode(step_results)
        _end_time = time.time()
        _time_taken = _end_time - _start_time
        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": final_x,
            "attention_mask": final_attention_mask,
            "positions": final_positions,
            "history": history,
            "loss": None,
            "time_taken": [_time_taken] * len(out),
        }

    @torch.inference_mode()
    def generate(self, prompts: List[str]) -> List[str]:
        # Convert prompts to input batch
        token_ids = self.tokenizer(prompts, add_special_tokens=False)[
            "input_ids"
        ]

        batch = cast(
            ILMSeq2SeqPredictionBatch,
            prepare_prefix_ids(
                token_ids,
                self.tokenizer.pad_token_id,
                max_seq_len=None,
                cls_token_id=self.tokenizer.cls_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                bos_side="left",
            ),
        )

        batch["target_ids"] = None
        constraint = torch.ones_like(batch["attention_mask"], dtype=torch.bool)
        constraint[:, -1] = False
        batch["constraint"] = constraint
        cls_position = batch.get(
            "cls_position",
            torch.zeros(
                batch["input_ids"].shape[0],
                dtype=torch.long,
                device=batch["input_ids"].device,
            ),
        )

        constraint = constraint.scatter(-1, cls_position.unsqueeze(-1), 1)
        batch["cls_position"] = cls_position
        if flags.DEBUG_PRINT_PREDS:
            print_batch_ilm(batch, "predict", self.tokenizer)
        # move to device
        batch = {
            k: v.to("cuda")
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
        preds = self.predict(batch)
        return preds["text"]


class ILMPredictorWithStoppingClassification(
    ILMPredictorWithLengthClassification
):
    def __init__(
        self,
        *args,
        stopping_threshold: float = 0.9,
        **kwargs,
    ):
        super().__init__(
            *args, stopping_threshold=stopping_threshold, **kwargs
        )


# endregion: Predictors
###############################################################
