from typing import (
    Literal,
    Optional,
    cast,
)

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT

from .nn import (
    masked_ce_last_two_dims,
)

from .datamodule_ilm import (
    Tokenizer,
)
from .types_ilm import (
    ILMBatch,
)
from xlm.harness import (
    Harness,
    LossFunction,
)
from xlm.utils.rank_zero import RankedLogger
from .types_ilm import (
    ILMModel,
    ILMLossDict,
)

logger = RankedLogger(__name__, rank_zero_only=True)


###############################################################
# region: Loss functions


class ILMLossWithMaskedCE(LossFunction[ILMBatch, ILMLossDict]):
    def __init__(
        self,
        model: Optional[ILMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
        length_loss: Optional[Literal["binary_ce", "ce"]] = None,
        length_loss_weight: Optional[float] = None,
        stopping_class_weight: Optional[float] = None,
        # TODO: Remove these arguments.
        loss_on_padding: bool = False,
        use_constraint: bool = False,
        input_constraint: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer  # type: ignore
        self._min_value: Optional[float] = None
        self.mask_token_id_tensor = None
        self.length_loss = length_loss
        self._length_loss_weight: Optional[float] = length_loss_weight
        self._stopping_class_weight: Optional[float] = stopping_class_weight
        # configure these in configure
        self.stopping_class_weight: Optional[TT] = None
        self.length_loss_weight: Optional[TT] = None
        if stopping_class_weight is not None:
            if length_loss != "binary_ce":
                raise ValueError(
                    "stopping_class_weight is only supported when length_loss is 'binary_ce'"
                )
        if input_constraint:
            raise NotImplementedError("input_constraint is not supported")
        if loss_on_padding:
            raise ValueError("loss_on_padding is not supported")
        if use_constraint:
            raise NotImplementedError("use_constraint is not supported")

    def min_value(self, logits) -> float:
        if self._min_value is None:
            self._min_value = torch.finfo(logits.dtype).min
        return self._min_value

    def configure(self, pl_module: Harness):
        self.mask_token_id_tensor = torch.tensor(  # type: ignore
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )
        if self._stopping_class_weight is not None:
            if not (0.0 <= self._stopping_class_weight <= 1.0):
                raise ValueError(
                    f"stopping_class_weight must be between 0 and 1, got {self._stopping_class_weight}"
                )
            self.stopping_class_weight = torch.tensor(
                [
                    self._stopping_class_weight,
                    1.0 - self._stopping_class_weight,
                ],
                device=pl_module.device,
            )

        if self._length_loss_weight is not None:
            if not (0.0 <= self._length_loss_weight <= 1.0):
                raise ValueError(
                    f"length_loss_weight must be between 0 and 1, got {self._length_loss_weight}"
                )
            self.length_loss_weight = torch.tensor(
                self._length_loss_weight,
                device=pl_module.device,
            )
        else:
            self.length_loss_weight = torch.tensor(
                1.0, device=pl_module.device
            )

    def create_mask(
        self,
        non_drop_non_pad: Bool[TT, " batch seq_len"],
        cls_position: Optional[Integer[TT, " batch"]],
        constraint: Optional[Bool[TT, " batch seq_len"]],
    ) -> Bool[TT, " batch seq_len"]:
        if cls_position is not None:
            # mask out the cls position
            _constraint = non_drop_non_pad.scatter(
                -1, cls_position.unsqueeze(-1), 0
            ).logical_not()
        else:
            _constraint = non_drop_non_pad.logical_not_()
            _constraint[:, 0] = 1

        if constraint is None or not self.use_constraint:
            return _constraint.unsqueeze(-1)

        elif self.use_constraint and constraint is not None:
            return _constraint.logical_or_(constraint).unsqueeze(-1)
        else:
            raise ValueError("Invalid constraint")

    def __call__(
        self,
        batch: ILMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMLossDict:
        sparse_target = batch["target_ids"]
        batch["target_ids"] = sparse_target.to_dense()
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        batch["target_ids"] = sparse_target
        return loss_dict

    def loss_fn(
        self,
        batch: ILMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMLossDict:

        x, attention_mask, target_ids, n_drops, cls_position = (
            batch["input_ids"],  # shape (batch, post_seq_len)
            batch["attention_mask"],  # shape (batch, post_seq_len)
            batch["target_ids"],  # shape (batch, post_seq_len, vocab_size)
            batch["n_drops"],  # shape (batch, post_seq_len)
            batch["cls_position"],  # shape (batch,)
        )
        # TODO (efficiency): This might be redundant if the data pipeline is aware of loss_on_padding flag.
        positions = (torch.cumsum(attention_mask, dim=-1) - 1).clamp(
            min=0
        )  # shape (batch, seq_len)
        model = cast(ILMModel, self.model)
        logits, length_logits = model(
            x,
            attention_mask,
            positions=positions,
            cls_position=cls_position,
        )
        total_n_drops = n_drops.sum(dim=-1)
        p_target = target_ids / (total_n_drops[:, None, None] + 1e-9)
        loss_mask = self.create_mask(
            attention_mask, cls_position, None
        )  # shape (batch, seq_len, vocab_size)
        ce = masked_ce_last_two_dims(
            logits,
            p_target,
            loss_mask,
            min_value=self.min_value(logits),
            inplace=False,
        )  # shape (batch,)
        example_loss = ce
        loss = example_loss.mean()

        assert self.length_loss_weight is not None
        if self.length_loss == "ce":
            length_logits = cast(Float[TT, " batch max_length"], length_logits)
            length_loss = (
                self.length_loss_weight
                * torch.nn.functional.cross_entropy(
                    length_logits, total_n_drops, reduction="none"
                )
            )  # shape (batch,)
        elif self.length_loss == "binary_ce":
            length_logits = cast(Float[TT, " batch 2"], length_logits)
            length_loss = (
                self.length_loss_weight
                * torch.nn.functional.cross_entropy(
                    length_logits,
                    (total_n_drops > 0).long(),
                    weight=self.stopping_class_weight,
                    reduction="none",
                )
            )  # shape (batch,)
        else:
            raise ValueError(f"Invalid length_loss: {self.length_loss}")
        example_loss = ce + length_loss
        loss = example_loss.mean()

        return {
            "loss": loss,
            "batch_loss": example_loss.detach(),
            "per_example_length_loss": length_loss.detach(),
            "per_example_ce": ce.detach(),
            "length_logits": length_logits.detach(),
            "n_drops": n_drops.detach(),
        }
