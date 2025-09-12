from numpy import logical_and
import torch
from typing import Any, Dict, Optional, cast
from xlm.harness import Harness, LossFunction
from xlm.utils.nn import masked_mean
from xlm.utils.rank_zero import RankedLogger
from xlm.lm.indigo.types_indigo import (
    IndigoBatch,
    IndigoLossDict,
    IndigoModelProtocol,
)
from xlm.lm.indigo.utils import (
    get_absolute_position_matrix,
    get_tertiary_relative_position_matrix,
    get_left_right_pointer_position,
    masked_logsumexp,
)

logger = RankedLogger(__name__, rank_zero_only=True)


class IndigoLoss(LossFunction[IndigoBatch, IndigoLossDict]):
    """
    Indigo Loss = word prediction CE + position prediction CE

    Each term is computed per step and masked. We report total loss,
    individual components, perplexity, and accuracies.
    """

    def __init__(
        self, model=None, tokenizer=None, position_loss_weight: float = 1.0
    ):
        """Initialize the Indigo loss function.

        Args:
            model: The Indigo model instance
            tokenizer: Tokenizer with cls_token_id and pad_token_id
            position_loss_weight: Weight for position loss component
        """
        self.model: IndigoModelProtocol = model
        self.tokenizer = tokenizer  # type: ignore
        self._min_value: Optional[float] = (
            None  # will be configured in the first call to loss_fn
        )
        self.ignore_positions: Optional[torch.Tensor] = None
        self.position_loss_weight = position_loss_weight

    def configure(self, pl_module: Harness):
        if (
            self.tokenizer is None
            or self.tokenizer.cls_token_id is None
            or self.tokenizer.pad_token_id is None
        ):
            raise ValueError(
                "tokenizer must be provided and have cls_token_id and pad_token_id"
            )
        # Create ignore_positions tensor on the correct device
        self.ignore_positions = torch.tensor(
            [-100, self.tokenizer.cls_token_id, self.tokenizer.pad_token_id],
            device=pl_module.device,  # type: ignore
            dtype=torch.long,
        )

    def __call__(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.loss_fn(batch)  # type: ignore

    def min_value(self, logits) -> float:
        if self._min_value is None:
            self._min_value = torch.finfo(logits.dtype).min
        return self._min_value

    def loss_fn(self, batch: IndigoBatch) -> IndigoLossDict:
        assert self.model is not None, "model must be attached before training"

        # Inputs
        input_ids = batch["input_ids"]  # (bs, seq_len)
        attention_mask = batch["attention_mask"]  # (bs, seq_len)
        target_ids = batch["target_ids"]  # (bs, seq_len)
        pi = batch["pi"]  # (bs, seq_len)
        # Create causal attention mask with padding consideration
        _, seq_len = attention_mask.shape
        causal_attention_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=attention_mask.device,
            )
        )
        # Expand attention_mask to (batch, query_len(1), key_len) and combine with causal mask
        expanded_attention_mask = attention_mask.unsqueeze(
            1
        )  # (batch, 1, seq_len)
        causal_attention_mask = (
            expanded_attention_mask & causal_attention_mask
        )  # (batch, seq_len, seq_len)

        # forward pass
        hidden_states, vocab_logits = self.model(
            x_t=input_ids,
            pi=pi,
            attention_mask=causal_attention_mask,
        )  # (bs, seq_len, d_model), (bs, seq_len, vocab_size)

        # token loss
        # Transpose logits for cross_entropy (expects [N, C, ...] format)
        logits_T = vocab_logits.transpose(1, 2)
        # Compute cross-entropy loss ( ignores -100 positions)
        token_ce_loss = torch.nn.functional.cross_entropy(
            logits_T, target_ids, reduction="mean", ignore_index=-100
        )  # (batch, seq_len)

        # region: pointer loss
        # target_ids contain -100 so we need to be carful while sending in those targets
        # ideally it should not matter because we will ingore those target positions anyway
        # but -100 can still cause indexing errors for some datasets
        dummy_target_ids = target_ids.clone()
        dummy_target_ids[dummy_target_ids == -100] = 0
        # Obtain logits for lpl, and rpl of shape (bs, key_seq_len, query_seq_len) each.
        position_logits = self.model.get_position_logits(
            hidden_states, dummy_target_ids
        )  # shape (bs, key_seq_len, 2, query_seq_len)
        # perform masking on the logits, mask out future positions' logits, ie, set lower-triangular elements to -inf
        mask = torch.ones_like(
            position_logits, dtype=torch.bool
        )  # (bs, key_seq_len, 2, query_seq_len)
        # 1. keys can't be in the future, make upper traingular
        # 2. identify positions for which we don't predict pointers like EOD and PAD tokens
        if self.ignore_positions is None:
            raise RuntimeError(
                "Loss function not properly configured. Call configure() first."
            )
        ignore = torch.isin(target_ids, self.ignore_positions).unsqueeze(
            -2
        )  # (bs, 1, query_seq_len)
        mask[..., 0, :].tril_(diagonal=-1).logical_or_(ignore)
        mask[..., 1, :].tril_(diagonal=-1).logical_or_(ignore)
        # min_value = self.min_value(position_logits)
        masked_position_logits = position_logits.masked_fill(
            mask, float("-inf")
        )
        lse = torch.logsumexp(
            masked_position_logits.reshape(
                mask.size(0),
                -1,
                mask.size(
                    -1
                ),  # (bs, 2*key_seq_len, query_seq_len) # note that this is an interleaved logits matrix and cannot be split
            ),
            dim=-2,
        )  # (bs, query_seq_len)
        # index using target_ids
        # We will get -inf in lse for PAD and EOD query positions. We will pluck them out later before computing the loss.
        lp, rp = get_left_right_pointer_position(
            pi,
        )  # (bs, query_seq_len), (bs, query_seq_len)

        # logits for ground truth pointers
        left_logits = (
            position_logits[:, :, 0]  # (bs, key_seq_len, query_seq_len)
            .gather(dim=-2, index=lp.unsqueeze(-2))
            .squeeze(-2)
        )  # (bs, query_seq_len)
        right_logits = (
            position_logits[:, :, 1]
            .gather(dim=-2, index=rp.unsqueeze(-2))
            .squeeze(-2)
        )  # (bs, query_seq_len)
        lae = torch.logaddexp(left_logits, right_logits)  # (bs, query_seq_len)
        positions_to_compute_pos_loss = (
            mask.reshape(mask.size(0), -1, mask.size(-1)).all(dim=-2)
        ).logical_not()  # (bs, query_seq_len)
        position_loss = (
            lse[positions_to_compute_pos_loss]
            - lae[positions_to_compute_pos_loss]
        ).mean()
        total_loss = token_ce_loss + self.position_loss_weight * position_loss
        return {
            "loss": total_loss,
            "token_loss": token_ce_loss.detach(),
            "position_loss": position_loss.detach(),
        }


if __name__ == "__main__":
    torch.manual_seed(42)
    d_model = 2
    seq_len = 3
    vocab_size = 11
    effective_seq_len = seq_len + 3
    bs = 2

    # BOS, EOS, EOD = 0, 1, 2, PAD = 10
    # fmt: off
    pre_input_ids = torch.tensor(
        [
            [0, 1, 4, 5, 6, 2, 10, 10], # last two tokens are PAD
            [0, 1, 7, 8, 9, 7, 4,  2],
        ]
    ) # the joint permuted sequence
    input_ids = pre_input_ids[:, :-1]
    pi = torch.tensor(
        [
            [0, 4, 3, 1, 2, 5, 6],
            [0, 5, 3, 4, 2, 1, 6],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
    )
    target_ids = pre_input_ids[:, 1:]
    # target_ids will be:
    # torch.tensor(
    #    [
    #        [1, 4, 5, 6, 2, 10, 10],
    #        [1, 7, 8, 9, 2, 7, 4],
    #    ]
    # )
    # fmt: on
    class DummyModel:
        def __call__(self, x_t, rel_matrix, attention_mask):
            _bs, _seq_len = x_t.shape
            return torch.rand(_bs, _seq_len, d_model), torch.rand(
                _bs, _seq_len, vocab_size
            )

        @property
        def device(self):
            return torch.device("cpu")

        def get_position_logits(self, hidden_states, target_ids):
            _bs, _seq_len = target_ids.shape
            return torch.rand(_bs, _seq_len, 2, _seq_len)

    class DummyTokenizer:
        cls_token_id = 2
        pad_token_id = 10

    model = DummyModel()
    tokenizer = DummyTokenizer()
    loss = IndigoLoss(model=model, tokenizer=tokenizer)
    loss.configure(model)
    loss_dict = loss(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "pi": pi,
        }
    )
    print(loss_dict)
