from typing import (
    Any,
    Dict,
    Optional,
)


from xlm.harness import (
    Harness,
    LossFunction,
)
from xlm.utils.rank_zero import RankedLogger
from .types_indigo import IndigoBatch, IndigoLossDict

logger = RankedLogger(__name__, rank_zero_only=True)


###############################################################
# region: Loss functions


class IndigoLoss(LossFunction[IndigoBatch, IndigoLossDict]):
    def __init__(
        self,
        model=None,
        tokenizer=None,
    ):
        self.model = model
        self.tokenizer = tokenizer  # type: ignore

    def configure(self, pl_module: Harness):
        pass  # nothing to do here for now

    def __call__(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.loss_fn(batch, batch_idx, dataloader_idx, dataloader_name)

    def loss_fn(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        # TODO (URV): Implement the loss function.

        # 1. Unpack inputs
        input_ids = batch["input_ids"]                       
        attention_mask = batch["attention_mask"]            
        target_token_ids = batch["target_token_ids"]         
        target_positions = batch["target_positions"]         
    
        assert self.model is not None
        

        # 2. Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        token_logits = outputs["token_logits"]               
        position_logits = outputs["position_logits"]        

        # 3. Compute token loss
        token_loss = torch.nn.functional.cross_entropy(
            token_logits,
            target_token_ids,
            ignore_index=-100,
            reduction="mean"
        )

        # 4. Compute position loss
        position_loss = torch.nn.functional.cross_entropy(
            position_logits,
            target_positions,
            ignore_index=-100,
            reduction="mean"
        )

        # 5. Combine losses
        total_loss = token_loss + position_loss

        return {
            "loss": total_loss,
            "token_loss": token_loss.detach(),
            "position_loss": position_loss.detach(),
        }
